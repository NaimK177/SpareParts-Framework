import time
from typing import Union
import concurrent.futures
import numpy as np
from NewEnvironment import Inventory, InventoryRS
from Policies import BaseStockPolicy, ProBSP
from sb3_contrib import MaskablePPO as PPO
from MaskedDQN import MaskedDoubleDQN as DQN


def run_replication(env:Union[Inventory, InventoryRS], policy:Union[BaseStockPolicy, ProBSP, PPO, DQN],
                    length:int, burn_in:int):

    """
    Run a single replication for the selected environment, and get the cost, average stock level, and the Fill rate

    :param env: The inventory environment instance (Inventory or InventoryRS)
    :param policy: The policy to be evaluated (BaseStockPolicy, ProBSP, PPO, or DQN)
    :param length: Number of steps to run in the replication
    :param burn_in: Number of initial steps to exclude from the cost calculation
    :return: Tuple containing (cost, expected stock level, fill rate, holding, ordering, emergency, order_size)
    """
    obs, info = env.reset()
    _cum_costs = 0
    for step in range(length + burn_in):
        if env.time_step == burn_in:
            _cum_costs = 0
        action, _states = policy.predict(obs, deterministic=True, action_masks=env.action_masks())
        obs, cost, terminated, truncated, info = env.step(action)
        _cum_costs += cost
    cost = info["average_cost"]
    holding_cost = info["holding_costs"]
    ordering_cost = info["ordering_costs"]
    emergency_cost = info["emergency_costs"]
    ES = info["average_inventory"]
    FR = info["fill_rate"]
    order_size = info["average_order_size"]
    return cost, ES, FR, holding_cost, ordering_cost, emergency_cost, order_size


def evaluate_policy(env: Union[Inventory, InventoryRS], policy: Union[BaseStockPolicy, ProBSP, PPO, DQN],
                    replication: int = 10, length:int = 20000, burn_in: int = 2000,
                    processors:int=1, cost_structure:bool=False):
    """
        Evaluates a given policy in the specified inventory environment.

        Args:
            env (Inventory): The inventory environment instance to evaluate the policy on.
            policy (Union[BaseStockPolicy, ProBSP]): The policy to be evaluated.
            replication (int, optional): The number of replications to run. Defaults to 10.
            length (int, optional): The number of steps in each replication. Defaults to 10000.
            burn_in (int, optional): The number of initial steps to exclude from evaluation. Defaults to 2000.
            processors (int, optional): The number of processors to use for parallel execution. Defaults to 2.

        Returns:
            tuple: A tuple containing:
                - avg_cost (float): The average cost across replications.
                - std_cost (float): The standard deviation of costs.
                - avg_FR (float): The average fill rate across replications.
                - std_FR (float): The standard deviation of fill rates.
                - avg_ES (float): The average expected stock across replications.
                - std_ES (float): The standard deviation of expected stock.

        Raises:
            Exception: If any replication fails during parallel execution.

        Notes:
            - If `processors` > 1, the function uses a `ProcessPoolExecutor` for parallel execution.
            - The burn-in period is used to stabilize the environment before evaluation.
            - Results are printed to the console, including average and standard deviation metrics.
    """
    cost_array = np.ones(replication) * np.nan
    FR_array = np.ones(replication) * np.nan
    ES_array = np.ones(replication) * np.nan
    holding_array = np.ones(replication) * np.nan
    ordering_array = np.ones(replication) * np.nan
    emergency_array = np.ones(replication) * np.nan
    order_size_array = np.ones(replication) * np.nan
    start_t = time.time()
    if processors > 1 and policy in [ProBSP, BaseStockPolicy, PPO]:
        with concurrent.futures.ProcessPoolExecutor(processors) as executor:
            futures = [executor.submit(run_replication, env, policy, length, burn_in) for repl in range(replication)]
            for repl, future in enumerate(futures):
                cost, ES, FR, hc, oc, ec, order_size = future.result()
                cost_array[repl] = cost
                ES_array[repl] = ES
                FR_array[repl] = FR
    else:
        for repl in range(replication):
            cost, ES, FR, hc, oc, ec, order_size = run_replication(env, policy, length, burn_in)
            cost_array[repl] = cost
            ES_array[repl] = ES
            FR_array[repl] = FR
            holding_array[repl] = hc
            ordering_array[repl] = oc
            emergency_array[repl] = ec
            order_size_array[repl] = order_size
    end_t = time.time()
    avg_cost = np.nanmean(cost_array)
    std_cost = np.nanstd(cost_array)
    avg_FR = np.nanmean(FR_array)
    std_FR = np.nanstd(FR_array)
    avg_ES = np.nanmean(ES_array)
    std_ES = np.nanstd(ES_array)

    print(f"{policy} - {env} - {replication} replications - {length} steps - {burn_in} burn-in period avg:")
    print(f"Costs={abs(avg_cost)} ± {std_cost:.4f} \t E[S]={avg_ES:.4f} ± {std_ES:.4f} \t "
          f"FR={avg_FR:.4f} ± {std_FR:.4f} \t Total Eval Time={end_t-start_t:.4f}")
    if cost_structure:
        print(f"Cost Structure:\n"
              f"Holding:{np.nanmean(holding_array):.4f} \t Ordering:{np.nanmean(ordering_array):.4f} \t "
              f"Emergency:{np.nanmean(emergency_array):.4f}")
        print(f"Average Order Size={np.nanmean(order_size_array)}")
    return abs(avg_cost), std_cost, avg_FR, std_FR, avg_ES, std_ES


def find_bsp(env: Inventory):
    """
    Find the Base Stock policy parameter for a given inventory instance
    :param env: The inventory environment instance.
    :return: the base stock level N
    """
    machines = env.num_machines
    bsp = 0
    best_bsp = 0
    best_cost = np.inf
    step = 1
    for iterations in range(machines + 1):
        cost = evaluate_policy(env, policy=BaseStockPolicy(env=env, bs_level=bsp, max_batch_size=env.max_batch_size),
                               replication=12, processors=1)[0]
        if cost < best_cost:
            best_cost = cost
            best_bsp = bsp
            bsp += step
            if bsp > machines:
                break
        else:
            break
    return best_bsp, best_cost


def find_xo(env:Inventory, xo_start=0, initial_inventory=0):
    """
    Find the best ordering threshold (xo) for an inventory environment given an initial stock level.

    Args:
        env (Inventory): The inventory environment instance.
        xo_start (int, optional): Initial value for the ordering threshold. Defaults to 0.
        initial_inventory (int, optional): Initial stock level. Defaults to 0.

    Returns: tuple: A tuple containing:
        - best_xo (int): The optimal ordering threshold value.
        - best_cost (float): The minimum cost achieved with the optimal threshold.
    """
    inventory = env
    xo = xo_start
    step = -5
    best_xo = xo_start
    best_cost = np.inf
    iter_best_xo = xo_start
    iter_old_best_xo = xo_start
    iter_best_cost = np.inf
    iter_old_best_cost = np.inf
    for iterations in range(30):
        cost = evaluate_policy(inventory, policy=ProBSP(env=inventory, xo=xo, n=initial_inventory,
                                                        max_batch_size=env.max_batch_size),
                               replication=12, processors=1)[0]
        if cost < iter_best_cost:
            iter_old_best_xo = iter_best_xo
            iter_old_best_cost = iter_best_cost
            iter_best_xo = xo
            iter_best_cost = cost
            if cost < best_cost:
                best_xo = xo
                best_cost = cost
            xo = xo + step
        elif cost > iter_best_cost:
            step /= 2
            iter_best_cost = iter_old_best_cost
            xo = iter_old_best_xo + step
        if abs(step) < 1:
            break
    return best_xo, best_cost


def find_probsp(env: Inventory):
    """
        Finds the optimal ProBSP (Probabilistic Base Stock Policy) parameters for the given inventory environment.

        This function determines the best combination of `n` (initial inventory level) and `xo` (order-up-to level)
        to minimize the cost in the inventory environment. It first evaluates the Base Stock Policy (BSP) to find
        the baseline cost and then iteratively adjusts the ProBSP parameters to improve upon the BSP cost.

        Args:
            env (Inventory): The inventory environment instance to optimize the ProBSP parameters for.

        Returns:
            tuple: A tuple containing:
                - best_n (int): The optimal initial inventory level.
                - best_xo (int): The optimal order-up-to level.
                - best_cost (float): The minimum cost achieved with the optimal parameters.

        Notes:
            - If the BSP level is 0, the function directly optimizes `xo` for ProBSP.
            - If the BSP level is greater than 0, the function iteratively reduces `n` and optimizes `xo` for each level.
            - Early stopping is applied if reducing `n` results in worse performance.
        """
    bsp_level, bsp_cost = find_bsp(env=env)
    print(f"BSP={bsp_level}")
    xo = 100
    if bsp_level == 0:
        xo, xo_cost = find_xo(env=env, xo_start=xo, initial_inventory=bsp_level)
        return 0, xo, xo_cost
    else:
        best_cost = bsp_cost
        best_xo = xo
        best_n = bsp_level
        for n in reversed(range(bsp_level)):
            xo, xo_cost = find_xo(env=env, xo_start=xo, initial_inventory=n)
            if xo_cost < best_cost:
                best_cost = xo_cost
                best_xo = xo
                best_n = n
            else:
                print("Early stoppage because for less N, the ProBSP was worse")
                break
        return best_n, best_xo, best_cost