import logging
from typing import Dict, Tuple, List
import numpy as np
from tqdm import tqdm

import ase
import ase.io


from .data_utils import (
    load_from_xyz,
    load_from_xyz_becs,
    random_train_valid_split,
    graph_from_configuration,
    graph_from_configuration_becs,
    GraphDataLoader,
    load_config,
)


def datasets(
    *,
    r_max: float,
    config_dataset: dict,
) -> Tuple[
    GraphDataLoader,
    GraphDataLoader,
    GraphDataLoader,
    Dict[int, float],
    float,
]:
    """Load training and test dataset from xyz file"""
    
    train_path = load_config(config_dataset,'train_path',None)
    valid_path = load_config(config_dataset,'valid_path',None)
    test_path = load_config(config_dataset,'test_path',None)
    train_num = load_config(config_dataset,'train_num',None)
    valid_num = load_config(config_dataset,'valid_num',None)
    valid_fraction = load_config(config_dataset,'valid_fraction',None)
    test_num = load_config(config_dataset,'test_num',None)
    
    config_type_weights = load_config(config_dataset,'config_type_weights')
    seed = load_config(config_dataset,'seed',1234)
    loader_seed= load_config(config_dataset,'loader_seed',5678)
    energy_key= load_config(config_dataset,'energy_key',"energy")
    forces_key= load_config(config_dataset,'forces_key',"forces")
    n_node = load_config(config_dataset,'n_node',1)
    n_edge = load_config(config_dataset,'n_edge',1)
    n_graph = load_config(config_dataset,'n_graph',1)
    min_n_node = load_config(config_dataset,'min_n_node',1)
    min_n_edge = load_config(config_dataset,'min_n_edge',1)
    min_n_graph = load_config(config_dataset,'min_n_graph',1)
    n_mantissa_bits = load_config(config_dataset,'n_mantissa_bits',1)
    prefactor_stress = load_config(config_dataset,'prefactor_stress',1.0)
    remap_stress= load_config(config_dataset, 'remap_stress',None)

    print('nums check',n_node,n_edge,n_graph)

    atomic_energies_dict_xyz, all_train_configs = load_from_xyz(
        file_or_path=train_path,
        config_type_weights=config_type_weights,
        energy_key=energy_key,
        forces_key=forces_key,
        extract_atomic_energies=False,
        num_configs=train_num,
        prefactor_stress=prefactor_stress,
        remap_stress=remap_stress,
    )
    #logging.info(
    #    f"Loaded {len(all_train_configs)} training configurations from '{train_path}'"
    #)
    print(
        f"Loaded {len(all_train_configs)} training configurations from '{train_path}'"
    )

    if valid_path is not None:
        _, valid_configs = load_from_xyz(
            file_or_path=valid_path,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            extract_atomic_energies=False,
            num_configs=valid_num,
            prefactor_stress=prefactor_stress,
            remap_stress=remap_stress,
        )
        print(
            f"Loaded {len(valid_configs)} validation configurations from '{valid_path}'"
        )
        train_configs = all_train_configs
    elif valid_fraction is not None:
        print(
            f"Using random {100 * valid_fraction}% of training set for validation"
        )
        train_configs, valid_configs = random_train_valid_split(
            all_train_configs, int(len(all_train_configs) * valid_fraction), seed
        )
    elif valid_num is not None:
        print(f"Using random {valid_num} configurations for validation")
        train_configs, valid_configs = random_train_valid_split(
            all_train_configs, valid_num, seed
        )
    else:
        print("No validation set")
        train_configs = all_train_configs
        valid_configs = []
    del all_train_configs

    if test_path is not None:
        _, test_configs = load_from_xyz(
            file_or_path=test_path,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            extract_atomic_energies=False,
            num_configs=test_num,
            prefactor_stress=prefactor_stress,
            remap_stress=remap_stress,
        )
        print(
            f"Loaded {len(test_configs)} test configurations from '{test_path}'"
        )
    else:
        test_configs = []

    print(
        f"Total number of configurations: "
        f"train={len(train_configs)}, "
        f"valid={len(valid_configs)}, "
        f"test={len(test_configs)}"
    )

    train_loader = GraphDataLoader(
        graphs=[
            graph_from_configuration(c, cutoff=r_max) for c in tqdm(train_configs)
        ],
        n_node=n_node,
        n_edge=n_edge,
        n_graph=n_graph,
        min_n_node=min_n_node,
        min_n_edge=min_n_edge,
        min_n_graph=min_n_graph,
        n_mantissa_bits=n_mantissa_bits,
        shuffle=True,
        return_remainder=False,
        loader_seed = loader_seed,
    )
    valid_loader = GraphDataLoader(
        graphs=[
            graph_from_configuration(c, cutoff=r_max) for c in tqdm(valid_configs)
        ],
        n_node=n_node,
        n_edge=n_edge,
        n_graph=n_graph,
        min_n_node=min_n_node,
        min_n_edge=min_n_edge,
        min_n_graph=min_n_graph,
        n_mantissa_bits=n_mantissa_bits,
        shuffle=False,
        return_remainder=True,
        #loader_seed = loader_seed,
    )
    test_loader = GraphDataLoader(
        graphs=[
            graph_from_configuration(c, cutoff=r_max) for c in tqdm(test_configs)
        ],
        n_node=n_node,
        n_edge=n_edge,
        n_graph=n_graph,
        min_n_node=min_n_node,
        min_n_edge=min_n_edge,
        min_n_graph=min_n_graph,
        n_mantissa_bits=n_mantissa_bits,
        return_remainder=True,
        shuffle=False,
        # loader_seed = loader_seed,
    )
    return train_loader, valid_loader, test_loader,  r_max # atomic_energies_dict_xyz





def datasets_groups(
    valid_paths,
    r_max: float,
    config_type_weights: Dict = None,
    train_num: int = None,
    valid_path: str = None,
    valid_fraction: float = None,
    valid_num: int = None,
    test_path: str = None,
    test_num: int = None,
    seed: int = 1234,
    loader_seed: int = 5678,
    energy_key: str = "energy",
    forces_key: str = "forces",
    n_node: int = 1,
    n_edge: int = 1,
    n_graph: int = 1,
    min_n_node: int = 1,
    min_n_edge: int = 1,
    min_n_graph: int = 1,
    n_mantissa_bits: int = 1,
    prefactor_stress: float = 1.0,
    remap_stress: np.ndarray = None,
) -> List:
    """Load training and test dataset from xyz file"""
    
    valid_loaders = []
    
    for valid_path in valid_paths:
        valid_name = valid_path.split('/')[-1]   #.split('.')[0]
        _, valid_configs = data.load_from_xyz(
            file_or_path=valid_path,
            config_type_weights=config_type_weights,
            energy_key=energy_key,
            forces_key=forces_key,
            extract_atomic_energies=False,
            #num_configs=valid_num,
            prefactor_stress=prefactor_stress,
            remap_stress=remap_stress,
        )
        #print('vconfigs',valid_configs)

        valid_loader = GraphDataLoader(
            graphs=[
            data.graph_from_configuration(c, cutoff=r_max) for c in tqdm.tqdm(valid_configs)
            ],
            n_node=n_node,
            n_edge=n_edge,
            n_graph=n_graph,
            min_n_node=min_n_node,
            min_n_edge=min_n_edge,
            min_n_graph=min_n_graph,
            n_mantissa_bits=n_mantissa_bits,
            shuffle=False,
            #loader_seed = loader_seed,
        )
        
        valid_loaders.append((valid_name,valid_loader))
   
    return valid_loaders


def becs_eps_datasets(
    *,
    r_max: float,
    train_path: str,
    config_type_weights: Dict = None,
    train_num: int = None,
    valid_path: str = None,
    valid_fraction: float = None,
    valid_num: int = None,
    test_path: str = None,
    test_num: int = None,
    seed: int = 1234,
    loader_seed: int = 5678,
    n_node: int = 1,
    n_edge: int = 1,
    n_graph: int = 1,
    min_n_node: int = 1,
    min_n_edge: int = 1,
    min_n_graph: int = 1,
    n_mantissa_bits: int = 1,
    prefactor_stress: float = 1.0,
    remap_stress: np.ndarray = None,
) -> Tuple[
    GraphDataLoader,
    GraphDataLoader,
    GraphDataLoader,
    Dict[int, float],
    float,
]:
    
    #print('run load xyz becs')
    
    all_train_configs = load_from_xyz_becs(
        file_or_path=train_path,
        num_configs=train_num,
    )
    
    train_configs = all_train_configs
    
    if valid_path is not None:
        valid_configs = load_from_xyz_becs(
            file_or_path=valid_path,
            num_configs=valid_num,
        )
    elif valid_fraction is not None:
        logging.info(
            f"Using random {100 * valid_fraction}% of training set for validation"
        )
        train_configs, valid_configs = random_train_valid_split(
            train_configs, int(len(train_configs) * valid_fraction), seed
        )
    elif valid_num is not None:
        logging.info(f"Using random {valid_num} configurations for validation")
        train_configs, valid_configs = random_train_valid_split(
            train_configs, valid_num, seed
        )
    else:
        logging.info("No validation set")
        valid_configs = []
        
    train_loader = GraphDataLoader(
        graphs=[
            graph_from_configuration_becs(c, cutoff=r_max) for c in tqdm(train_configs)
        ],
        n_node=n_node,
        n_edge=n_edge,
        n_graph=n_graph,
        min_n_node=min_n_node,
        min_n_edge=min_n_edge,
        min_n_graph=min_n_graph,
        n_mantissa_bits=n_mantissa_bits,
        shuffle=True,
        return_remainder=False,
        loader_seed = loader_seed,
    )
    valid_loader = GraphDataLoader(
        graphs=[
            graph_from_configuration_becs(c, cutoff=r_max) for c in tqdm(valid_configs)
        ],
        n_node=n_node,
        n_edge=n_edge,
        n_graph=n_graph,
        min_n_node=min_n_node,
        min_n_edge=min_n_edge,
        min_n_graph=min_n_graph,
        n_mantissa_bits=n_mantissa_bits,
        shuffle=False,
        return_remainder=False,
        loader_seed = loader_seed,
    )
    test_loader = None
    return train_loader, valid_loader, test_loader, r_max


