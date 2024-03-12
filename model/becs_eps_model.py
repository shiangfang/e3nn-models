import functools
import math
from typing import Callable, Dict, Optional, Union, List
import numpy as np
import pickle


import e3nn_jax as e3nn
import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import numpy as np

import jax.numpy as jnp

from e3nn_jax import Irreps, Irrep
from e3nn_jax import IrrepsArray
from e3nn_jax import FunctionalLinear
from e3nn_jax.legacy import FunctionalFullyConnectedTensorProduct, FunctionalTensorProduct
from e3nn_jax.haiku import FullyConnectedTensorProduct, Linear


from jax.nn import initializers
from jax import tree_util
from jax import jit
from jax import vmap
from jax import tree_map


import operator
import jax.nn
import functools


from .utils import (
    create_directory_with_random_name,
    compute_avg_num_neighbors,
    bessel_basis,
    soft_envelope,
    safe_norm,
)

from .data_utils import (
    get_atomic_number_table_from_zs,
    compute_average_E0s,
)

from .blocks import (
    RadialEmbeddingBlock,
    LinearNodeEmbeddingBlock,
)

from .nequip_model import (
    NequIPConvolution,
)

partial = functools.partial


Array = jnp.ndarray

UnaryFn = Callable[[Array], Array]

f32 = jnp.float32




class BECS_EPS_nequip_base_Model(hk.Module):
    def __init__(
        self,
        *,
        #output_irreps : e3nn.Irreps,
        graph_net_steps: int,
        use_sc: bool,
        nonlinearities: Union[str, Dict[str, str]],
        hidden_irreps: str,
        max_ell: int = 3,
        num_basis: int = 8,
        r_max: float = 4.,
        num_species: int = None,
        avg_r_min: float = None,
        num_features: int = 7,
        radial_basis: Callable[[jnp.ndarray], jnp.ndarray],
        radial_net_nonlinearity: str = 'raw_swish',
        radial_net_n_hidden: int = 64,
        radial_net_n_layers: int = 2,
        radial_envelope: Callable[[jnp.ndarray], jnp.ndarray],
        shift: float = 0.,
        scale: float = 1.,
        avg_num_neighbors: float = 1.,
        scalar_mlp_std: float = 4.0,
    ):
        super().__init__()
        #print('calling Nequip init')
        
        #output_irreps = e3nn.Irreps(output_irreps)
        #self.output_irreps = output_irreps

        self.num_features = num_features
        self.max_ell=max_ell
        self.sh_irreps = e3nn.Irreps.spherical_harmonics(max_ell)
        self.r_max = r_max
        self.avg_num_neighbors = avg_num_neighbors
        self.hidden_irreps = Irreps(hidden_irreps)
        self.num_species = num_species
        self.graph_net_steps = graph_net_steps
        self.use_sc = use_sc
        self.nonlinearities = nonlinearities
        self.radial_net_nonlinearity = radial_net_nonlinearity
        self.radial_net_n_hidden = radial_net_n_hidden
        self.radial_net_n_layers = radial_net_n_layers
        self.num_basis = num_basis
        self.scalar_mlp_std = scalar_mlp_std
        
        
        #self.node_embedding = LinearNodeEmbeddingBlock(
        #    self.num_species, self.num_features * self.hidden_irreps
        #)
        self.node_embedding = LinearNodeEmbeddingBlock(
            self.num_species, self.hidden_irreps
        )
        
        #print('self node embed', self.node_embedding ) #,self.num_features * self.hidden_irreps)
        
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            avg_r_min=avg_r_min,
            basis_functions=radial_basis,
            envelope_function=radial_envelope,
        )
        
        
    def __call__(
        self,
        vectors: jnp.ndarray,  # [n_edges, 3]
        node_specie: jnp.ndarray,  # [n_nodes] int between 0 and num_species-1
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
    ) -> e3nn.IrrepsArray:
        r_max = jnp.float32(self.r_max)
        hidden_irreps = Irreps(self.hidden_irreps)
        
        edge_src = senders
        edge_dst = receivers
        embedding_irreps = Irreps(f'{self.num_species}x0e')
        
        node_attrs = self.node_embedding(node_specie).astype(
            vectors.dtype
        )
    

        # edge embedding
        lengths = safe_norm(vectors, axis=-1)
        
        #dR = vectors
        #scalar_dr_edge = space.distance(dR)
        #edge_sh = e3nn.spherical_harmonics(self.sh_irreps, vectors, normalize=False)
        edge_sh = e3nn.spherical_harmonics(self.sh_irreps,vectors / lengths[..., None],normalize=False,normalization="component")


        embedded_dr_edge = self.radial_embedding(lengths).array
        
        # embedding layer
        h_node = Linear(irreps_out=Irreps(hidden_irreps))(node_attrs)

        # convolutions
        for _ in range(self.graph_net_steps):
            h_node = NequIPConvolution(
                hidden_irreps=hidden_irreps,
                use_sc=self.use_sc,
                nonlinearities=self.nonlinearities,
                radial_net_nonlinearity=self.radial_net_nonlinearity,
                radial_net_n_hidden=self.radial_net_n_hidden,
                radial_net_n_layers=self.radial_net_n_layers,
                num_basis=self.num_basis,
                avg_num_neighbors=self.avg_num_neighbors,
                scalar_mlp_std=self.scalar_mlp_std
            )(h_node,
              node_attrs,
              edge_sh,
              edge_src,
              edge_dst,
              embedded_dr_edge
             )
        
        cg_L110 = e3nn.clebsch_gordan(1,1,0)
        cg_L111 = e3nn.clebsch_gordan(1,1,1)
        cg_L112 = e3nn.clebsch_gordan(1,1,2)
            
        h_node_becsL0 = Linear(irreps_out=Irreps('1x0e'))(h_node)
        h_node_becsL1 = Linear(irreps_out=Irreps('1x1e'))(h_node)
        h_node_becsL2 = Linear(irreps_out=Irreps('1x2e'))(h_node)
        
        h_node_becs = jnp.einsum('jki,ai->ajk',cg_L112,h_node_becsL2.array)
        h_node_becs += jnp.einsum('jki,ai->ajk',cg_L111,h_node_becsL1.array)
        h_node_becs += jnp.einsum('jki,ai->ajk',cg_L110,h_node_becsL0.array)
        
        h_node_epsL0 = Linear(irreps_out=Irreps('1x0e'))(h_node)
        h_node_epsL1 = Linear(irreps_out=Irreps('1x1e'))(h_node)
        h_node_epsL2 = Linear(irreps_out=Irreps('1x2e'))(h_node)
        
        h_node_eps = jnp.einsum('jki,ai->ajk',cg_L112,h_node_epsL2.array)
        h_node_eps += jnp.einsum('jki,ai->ajk',cg_L111,h_node_epsL1.array)
        h_node_eps += jnp.einsum('jki,ai->ajk',cg_L110,h_node_epsL0.array)
        
        h_node_denoising = Linear(irreps_out=Irreps('1x1o'))(h_node).array
        
        # return h_node, h_node_becs, h_node_eps
        return h_node_becs, h_node_eps,h_node_denoising
    
    
def BECS_EPS_model(
    *,
    r_max: float,
    atomic_energies_dict: Dict[int, float] = None,
    train_graphs: List[jraph.GraphsTuple] = None,
    initialize_seed: Optional[int] = None,
    scaling: Callable = None,
    atomic_energies: Union[str, np.ndarray, Dict[int, float]] = None,
    avg_num_neighbors: float = "average",
    avg_r_min: float = None,
    num_species: int = None,
    path_normalization="path",
    gradient_normalization="path",
    learnable_atomic_energies=False,
    radial_basis: Callable[[jnp.ndarray], jnp.ndarray] = bessel_basis,
    radial_envelope: Callable[[jnp.ndarray], jnp.ndarray] = soft_envelope,
    save_dir_name = None,
    reload = None,
    **kwargs,
):
    if reload is None:
        becs_eps_model_setup = {}
        
        if train_graphs is None:
            z_table = None
        else:
            z_table = get_atomic_number_table_from_zs(
                z for graph in train_graphs for z in graph.nodes.species
            )
        print(f"z_table= {z_table}")
        
        becs_eps_model_setup['z_table'] = z_table

        if avg_num_neighbors == "average":
            avg_num_neighbors = compute_avg_num_neighbors(train_graphs)
            print(
                f"Compute the average number of neighbors: {avg_num_neighbors:.3f}"
            )
        else:
            print(f"Use the average number of neighbors: {avg_num_neighbors:.3f}")
            
        becs_eps_model_setup['avg_num_neighbors'] = avg_num_neighbors

        if avg_r_min == "average":
            avg_r_min = compute_avg_min_neighbor_distance(train_graphs)
            print(f"Compute the average min neighbor distance: {avg_r_min:.3f}")
        elif avg_r_min is None:
            print("Do not normalize the radial basis (avg_r_min=None)")
        else:
            print(f"Use the average min neighbor distance: {avg_r_min:.3f}")

        becs_eps_model_setup['avg_r_min'] = avg_r_min
        
        if save_dir_name:
            with open(f"{save_dir_name}/becs_eps_model_setup.pkl", "wb") as f:
                pickle.dump(becs_eps_model_setup, f) #  E0s.tolist()
        
    else:
        with open(f"{reload}/becs_eps_model_setup.pkl", "rb") as f:
            becs_eps_model_setup = pickle.load(f)
            
        if save_dir_name:
            with open(f"{save_dir_name}/becs_eps_model_setup.pkl", "wb") as f:
                pickle.dump(becs_eps_model_setup, f) #  E0s.tolist()
        
        z_table = becs_eps_model_setup['z_table']
        avg_num_neighbors = becs_eps_model_setup['avg_num_neighbors']
        avg_r_min = becs_eps_model_setup['avg_r_min']
        
        

    # check that num_species is consistent with the dataset
    if z_table is None:
        if train_graphs is not None:
            for graph in train_graphs:
                if not np.all(graph.nodes.species < num_species):
                    raise ValueError(
                        f"max(graph.nodes.species)={np.max(graph.nodes.species)} >= num_species={num_species}"
                    )
    else:
        if max(z_table.zs) >= num_species:
            raise ValueError(
                f"max(z_table.zs)={max(z_table.zs)} >= num_species={num_species}"
            )

    kwargs.update(
        dict(
            r_max=r_max,
            avg_num_neighbors=avg_num_neighbors,
            avg_r_min=avg_r_min,
            num_species=num_species,
            radial_basis=radial_basis,
            radial_envelope=radial_envelope,
        )
    )
    print(f"Create BECS/EPS (NequIP-based) model with parameters {kwargs}")

    @hk.without_apply_rng
    @hk.transform
    def model_(
        vectors: jnp.ndarray,  # [n_edges, 3]
        node_z: jnp.ndarray,  # [n_nodes]
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
    ) -> jnp.ndarray:
        e3nn.config("path_normalization", path_normalization)
        e3nn.config("gradient_normalization", gradient_normalization)
        
        becs_eps = BECS_EPS_nequip_base_Model(**kwargs)

        if hk.running_init():
            print(
                "model: "
                f"hidden_irreps={becs_eps.hidden_irreps} "
                f"sh_irreps={becs_eps.sh_irreps} ",
            )

        contributions = becs_eps(
            vectors, node_z, senders, receivers
        )
        
        return contributions

    if initialize_seed is not None and reload is None:
        params = jax.jit(model_.init)(
            jax.random.PRNGKey(initialize_seed),
            jnp.zeros((1, 3)),
            jnp.array([16]),
            jnp.array([0]),
            jnp.array([0]),
        )
    elif reload is not None:
        with open(f"{reload}/params.pkl", "rb") as f:
            params = pickle.load(f)
    else:
        params = None

    return model_.apply, params, kwargs['graph_net_steps']   #, num_interactions





