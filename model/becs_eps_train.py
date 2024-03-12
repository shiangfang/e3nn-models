import itertools
import time
from typing import Any, Callable, Dict, Optional, Tuple
import pickle

import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
import tqdm

from .utils import (
    compute_mae,
    compute_rel_mae,
    compute_rmse,
    compute_rel_rmse,
    compute_q95,
    compute_c,
)

from .data_utils import (
    GraphDataLoader,
)


def train_becs_eps(
    model: Callable,
    params: Dict[str, Any],
    loss_fn: Any,
    train_loader: GraphDataLoader,
    gradient_transform: Any,
    optimizer_state: Dict[str, Any],
    steps_per_interval: int,
    ema_decay: Optional[float] = None,
):
    num_updates = 0
    ema_params = params

    print("Started training")

    @jax.jit
    def update_fn(
        params, optimizer_state, ema_params, num_updates: int, graph: jraph.GraphsTuple
    ) -> Tuple[float, Any, Any]:
        # graph is assumed to be padded by jraph.pad_with_graphs
        mask = jraph.get_graph_padding_mask(graph)  # [n_graphs,]
        
        loss, grad = jax.value_and_grad(
            lambda params: jnp.mean(loss_fn(graph, model(params, graph)) * mask)
        )(params)
        updates, optimizer_state = gradient_transform.update(
            grad, optimizer_state, params
        )
        params = optax.apply_updates(params, updates)
        if ema_decay is not None:
            decay = jnp.minimum(ema_decay, (1 + num_updates) / (10 + num_updates))
            ema_params = jax.tree_util.tree_map(
                lambda x, y: x * decay + y * (1 - decay), ema_params, params
            )
        else:
            ema_params = params
        return loss, params, optimizer_state, ema_params

    last_cache_size = update_fn._cache_size()

    def interval_loader():
        i = 0
        while True:
            for graph in train_loader:
                yield graph
                i += 1
                if i >= steps_per_interval:
                    return

    for interval in itertools.count():
        yield interval, params, optimizer_state, ema_params

        # Train one interval
        p_bar = tqdm.tqdm(
            interval_loader(),
            desc=f"Train interval {interval}",
            total=steps_per_interval,
        )
        for graph in p_bar:
            num_updates += 1
            start_time = time.time()
            loss, params, optimizer_state, ema_params = update_fn(
                params, optimizer_state, ema_params, num_updates, graph
            )
            loss = float(loss)
            p_bar.set_postfix({"loss": f"{loss:7.3f}"})

            if last_cache_size != update_fn._cache_size():
                last_cache_size = update_fn._cache_size()

                print("Compiled function `update_fn` for args:")
                print(f"- n_node={graph.n_node} total={graph.n_node.sum()}")
                print(f"- n_edge={graph.n_edge} total={graph.n_edge.sum()}")
                print(f"Outout: loss= {loss:.3f}")
                print(
                    f"Compilation time: {time.time() - start_time:.3f}s, cache size: {last_cache_size}"
                )
                




def BECS_EPS_train(
    model,
    params,
    optimizer_state,
    train_loader,
    valid_loader,
    #test_loader,
    gradient_transform,
    loss_fn,
    max_num_intervals: int,
    steps_per_interval: int,
    save_dir_name: str,
    *,
    patience: Optional[int] = None,
    eval_train: bool = False,
    eval_test: bool = False,
    log_errors: str = "PerAtomRMSE",
    **kwargs,
):
    lowest_loss = np.inf
    patience_counter = 0
    #loss_fn = loss_becs()
    start_time = time.perf_counter()
    total_time_per_interval = []
    eval_time_per_interval = []

    for interval, params, optimizer_state, ema_params in train_becs_eps(
        model=model,
        params=params,
        loss_fn=loss_fn,
        train_loader=train_loader,
        gradient_transform=gradient_transform,
        optimizer_state=optimizer_state,
        steps_per_interval=steps_per_interval,
        **kwargs,
    ):
        total_time_per_interval += [time.perf_counter() - start_time]
        start_time = time.perf_counter()

        try:
            import profile_nn_jax
        except ImportError:
            pass
        else:
            profile_nn_jax.restart_timer()

        last_interval = interval == max_num_intervals
        
        with open(f"{save_dir_name}/params.pkl", "wb") as f:
            pickle.dump(params, f)

        with open(f"{save_dir_name}/ema_params.pkl", "wb") as f:
            pickle.dump(ema_params, f)

        def eval_and_print(loader, mode: str):
            metrics_ = evaluate_becs_eps(
                model=model,
                params=ema_params,
                loss_fn=loss_fn,
                data_loader=loader,
                name=mode,
            )
            metrics_["mode"] = mode
            metrics_["interval"] = interval
            #logger.log(metrics_)
            
            error_becs = "mae_becs"


            def _(x: str):
                v: float = metrics_.get(x, None)
                return f"{v:.4f}"
                
            print(
                f"Interval {interval}: {mode}: "
                #f"loss={loss_:.4f}, "
                f"{error_becs}={_(error_becs)}, "
            )
            #return loss_
            return
        
        #if eval_train:
        eval_and_print(train_loader, "eval_train")
        #if eval_train:
        eval_and_print(valid_loader, "eval_valid")  
            
        
        #if eval_train or last_interval:
            
        #    eval_and_print(train_loader, "eval_train")

        #if (
        #    (eval_test or last_interval)
        #    and test_loader is not None
        #    and len(test_loader) > 0
        #):
        #    eval_and_print(test_loader, "eval_test")

        #if valid_loader is not None and len(valid_loader) > 0:
        #    loss_ = eval_and_print(valid_loader, "eval_valid")

        #    if loss_ >= lowest_loss:
        #        patience_counter += 1
        #        if patience is not None and patience_counter >= patience:
        #            logging.info(
        #                f"Stopping optimization after {patience_counter} intervals without improvement"
        #            )
        #            break
        #    else:
        #        lowest_loss = loss_
        #        patience_counter = 0

        eval_time_per_interval += [time.perf_counter() - start_time]
        avg_time_per_interval = np.mean(total_time_per_interval[-3:])
        avg_eval_time_per_interval = np.mean(eval_time_per_interval[-3:])

        print(
            f"Interval {interval}: Time per interval: {avg_time_per_interval:.1f}s, "
            f"among which {avg_eval_time_per_interval:.1f}s for evaluation."
        )

        if last_interval:
            break

    print("Training complete")
    return interval, ema_params



def evaluate_becs_eps(
    model: Callable,
    params: Any,
    loss_fn: Any,
    data_loader: GraphDataLoader,
    name: str = "Evaluation",
) -> Tuple[float, Dict[str, Any]]:
    total_loss = 0.0
    num_graphs = 0
    
    if hasattr(model, "_cache_size"):
        last_cache_size = model._cache_size()
    else:
        last_cache_size = None
        
    delta_becs_list = []
    becs_list = []
    
    delta_eps_list = []
    eps_list = []
        
    start_time = time.time()
    p_bar = tqdm.tqdm(data_loader, desc=name, total=data_loader.approx_length())
    for ref_graph in p_bar:
        output = model(params, ref_graph)
        pred_graph = ref_graph._replace(
            nodes=ref_graph.nodes._replace(becs=output["becs"]),
            globals=ref_graph.globals._replace(eps=output["eps"]),
        )

        if last_cache_size is not None and last_cache_size != model._cache_size():
            last_cache_size = model._cache_size()

            print("Compiled function `model` for args:")
            print(f"- n_node={ref_graph.n_node} total={ref_graph.n_node.sum()}")
            print(f"- n_edge={ref_graph.n_edge} total={ref_graph.n_edge.sum()}")
            print(f"cache size: {last_cache_size}")

        ref_graph = jraph.unpad_with_graphs(ref_graph)
        pred_graph = jraph.unpad_with_graphs(pred_graph)
    
    
        num_graphs += len(ref_graph.n_edge)
        
        if ref_graph.nodes.becs is not None:
            delta_becs_list.append(ref_graph.nodes.becs - pred_graph.nodes.becs)
            becs_list.append(ref_graph.nodes.becs)
        if ref_graph.globals.eps is not None:
            delta_eps_list.append(ref_graph.globals.eps - pred_graph.globals.eps)
            eps_list.append(ref_graph.globals.eps)
           
            

    if num_graphs == 0:
        print(f"No graphs in data_loader ! Returning 0.0 for {name}")
        #return 0.0, {}
        return {}
    
    aux = {
        #"loss": avg_loss,
        "time": time.time() - start_time,
        "mae_becs": None,
        "mae_eps" : None,
        #"rel_mae_becs": None,
        #"rmse_becs": None,
        #"rel_rmse_becs": None,
        #"q95_becs": None,
    }
    
    if len(delta_becs_list) > 0:
        delta_becs = np.concatenate(delta_becs_list, axis=0)
        becs = np.concatenate(becs_list, axis=0)
        aux.update(
            {
                # Mean absolute error
                "mae_becs": compute_mae(delta_becs),
                #"rel_mae_becs": compute_rel_mae(delta_becs, becs),
                # Root-mean-square error
                #"rmse_becs": compute_rmse(delta_becs),
                #"rel_rmse_becs": compute_rel_rmse(delta_becs, becs),
                # Q_95
                #"q95_becs": compute_q95(delta_becs),
            }
        )
    
    if len(delta_eps_list) > 0:
        delta_eps = np.concatenate(delta_eps_list, axis=0)
        eps = np.concatenate(eps_list, axis=0)
        aux.update(
            {
                # Mean absolute error
                "mae_eps": compute_mae(delta_eps),
                #"rel_mae_becs": compute_rel_mae(delta_becs, becs),
                # Root-mean-square error
                #"rmse_becs": compute_rmse(delta_becs),
                #"rel_rmse_becs": compute_rel_rmse(delta_becs, becs),
                # Q_95
                #"q95_becs": compute_q95(delta_becs),
            }
        )
    return aux
    #return avg_loss, aux
    
    
    
    
