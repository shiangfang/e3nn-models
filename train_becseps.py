import json
import os
import pickle
import random
import sys

from typing import Callable, Dict, List, Optional
import haiku as hk
import ase
import ase.io
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml


from model.datasets import becs_eps_datasets
from model.utils import (
    create_directory_with_random_name,
    compute_avg_num_neighbors,
)
from model.data_utils import (
    get_atomic_number_table_from_zs,
    compute_average_E0s,
)
from model.predictors import predict_becs_eps
from model.optimizer import optimizer
from model.becs_eps_train import BECS_EPS_train
from model.loss import BecsEpsLoss


from model.becs_eps_model import BECS_EPS_model

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)
np.set_printoptions(precision=3, suppress=True)


def main():
    with open(sys.argv[1]) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    save_dir_name = create_directory_with_random_name(
        os.path.splitext(os.path.basename(sys.argv[1]))[0]
    )
    
    # Save config and code
    with open(f"{save_dir_name}/config.yaml", "w") as f:
        yaml.dump(config, f)
    with open(f"{save_dir_name}/train.py", "w") as f:
        with open(sys.argv[0]) as g:
            f.write(g.read())
            
            
    train_loader, valid_loader,test_loader, r_max = becs_eps_datasets(
        r_max = config["cutoff"],
        train_path = config["dataset"]["train_path"],
        #valid_path = config["dataset"]["valid_path"],
        #train_num = config["dataset"]["train_num"],
        valid_num = config["dataset"]["valid_num"],
        n_node = config["dataset"]["num_nodes"],
        n_edge = config["dataset"]["num_edges"],
        n_graph = config["dataset"]["num_graphs"],
    )
    
    print(len(train_loader.graphs))
    print(len(valid_loader.graphs))
    #for tdata in train_loader:
    #    print(tdata.nodes.species)
    model_fn, params, num_message_passing = BECS_EPS_model(
        r_max=r_max,
        atomic_energies_dict={},
        train_graphs=train_loader.graphs,
        initialize_seed=config["model"]["seed"],
        num_species = config["model"]["num_species"],
        use_sc = True,
        graph_net_steps = config["model"]["num_layers"],
        hidden_irreps = config["model"]["internal_irreps"],
        nonlinearities =  {'e': 'swish', 'o': 'tanh'},
        save_dir_name = save_dir_name,
        reload = config["initialization"]['reload'] if 'reload' in config["initialization"] else None,
    )
    
    print("num_params:", sum(p.size for p in jax.tree_util.tree_leaves(params)))
    
    
    predictor = jax.jit(
        lambda w, g: predict_becs_eps(lambda *x: model_fn(w, *x), g)
    )
    
    gradient_transform, steps_per_interval, max_num_intervals = optimizer(
        lr = config["training"]["learning_rate"],
        max_num_intervals = config["training"]["max_num_intervals"],
        steps_per_interval = config["training"]["steps_per_interval"],
        # weight_decay = config["training"]["weight_decay"],
    )
    optimizer_state = gradient_transform.init(params)
    print("optimizer num_params:", sum(p.size for p in jax.tree_util.tree_leaves(optimizer_state)))
    
    loss_fn = BecsEpsLoss(
        becs_weight = config["training"]["becs_weight"],
        becs_sum_weight = config["training"]["becs_sum_weight"],
        eps_weight = config["training"]["eps_weight"],
    )
    
    BECS_EPS_train(
        predictor,
        params,
        optimizer_state,
        train_loader,
        valid_loader,
        #test_loader,
        gradient_transform,
        loss_fn =loss_fn,
        max_num_intervals = max_num_intervals,
        steps_per_interval = steps_per_interval,
        save_dir_name = save_dir_name,
        patience = config["training"]["patience"],
        #ema_decay = config["training"]["ema_decay"],
    )
    print('done!')
    
if __name__ == "__main__":
    main()

