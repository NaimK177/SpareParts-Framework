"""
This file runs experiments and write the result in the results.csv
It first checks whether the problem instance exists and solved. If not it runs the experiment.
The experiment is defined by:
 [num_machines, lead_time_p, mttf, a, co, ce, max_batch_size, sorted_degradation]
The results of an experiment are:
[avg_cost, std_cost, avg_FR, std_FR, avg_ES, std_ES]
We will test the BSP, ProBSP, DQN, and PPO in here.
A line in a csv will read as follows:
policy,env,num_machines,lead_time_p,mttf,a,co,ce,max_batch_size,sorted_degradation,avg_cost,std_cost,avg_FR,
std_FR,avg_ES,std_ES

policies are ['ppo', 'dqn', 'bsp', 'probsp', 'ppo_bsp', 'ppo_probsp', 'dqn_bsp', 'dqn_probsp']

"""
import csv
from itertools import product
import concurrent.futures
import pandas as pd
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib import MaskablePPO as PPO
from MaskedDQN import MaskedDoubleDQN as DDQN
from Auxiliaries import evaluate_policy, find_bsp, find_probsp
from NewEnvironment import InventoryRS, Inventory, GeometricOrderPipeline
from Policies import BaseStockPolicy, ProBSP

file_path = "results.csv"

policy = ['ppo', 'dqn', 'bsp', 'probsp', 'ppo_probsp', 'ppo_bsp', 'dqn_bsp', 'dqn_probsp']
env = [Inventory]
num_machines = [1, 2, 5, 10, 30]  # 5
p = [0.2, 0.5]  # 2
mttf = [5, 10, 20]  # 3
a = [1]
co = [2]
ce = [5, 10]  # 2
batch_size = [3, 5]  # 2

num_machines = [1, 5, 30]
policy = ['probsp']
ce=[5]
batch_size = [5]


combinations = list(product(policy, env, num_machines, p, mttf, a, co, ce, batch_size))
total_experiments = len(combinations)
print(f"Total Experiments = {total_experiments}")
exit("Hi")
with open(file_path, mode='r') as file:
    df = pd.read_csv(file)
file.close()
print(f"Already finished experiments = {len(df)}")

def check_solution(experiment_data:list):
    policy, env, num_machines, p, mttf, a, co, ce, batch_size = experiment_data
    with open(file_path, mode='r') as file:
        df = pd.read_csv(file)
    print(f"Already finished experiments = {len(df)}")
    file.close()
    avg_cost = df.loc[(df.policy == policy) & (df.num_machines == num_machines)
                       & (df.lead_time_p == p) & (df.mttf == mttf) & (df.a == a) & (df.co == co) & (df.ce == ce)
                       & (df.max_batch_size == batch_size)].avg_cost.values
    if len(avg_cost) == 1:
        return True
    else:
        return False


def find_parameters(experiment_data:list):
    policy, env, num_machines, p, mttf, a, co, ce, batch_size = experiment_data
    with open(file_path, mode='r') as file:
        df = pd.read_csv(file)
    file.close()
    _, rs_policy = policy.split('_')
    condition = ((df.policy == rs_policy) & (df.num_machines == num_machines)
                & (df.lead_time_p == p) & (df.mttf == mttf) & (df.a == a) & (df.co == co) & (df.ce == ce)
                & (df.max_batch_size == batch_size))
    n = df.loc[condition].n.values
    xo = df.loc[condition].xo.values
    return n, xo

def run_experiment(experiment_data:list):
    policy, env, num_machines, p, mttf, a, co, ce, batch_size = experiment_data
    if check_solution(experiment_data):
        print("Experiment Already Done - Skipping")
        return 0
    order_pipeline = GeometricOrderPipeline(num_machines, p)
    env = Inventory(num_machines, order_pipeline, max_batch_size=batch_size, mttf=mttf, a=a,
                    ordering_cost=co, emergency_cost=ce)
    # Initialize n, xo to 0 to write them in each row even when dqn or PPO are used
    n, xo = 0, 0
    if policy == 'ppo':
        pol = PPO(MaskableActorCriticPolicy, env=env, verbose=0)
        pol.learn(800000)
    elif policy == 'dqn':
        pol = DDQN(env, train_every=2, replay_memory_size=400000, update_target_every=10000)
        pol.learn(800000)
    elif policy == 'bsp':
        n, _ = find_bsp(env)
        pol = BaseStockPolicy(env, n, env.max_batch_size)
    elif policy == 'probsp':
        n, xo, _ = find_probsp(env)
        pol = ProBSP(env, n, xo, env.max_batch_size)
    elif policy == 'ppo_bsp':
        n, xo = find_parameters(experiment_data)
        env = InventoryRS(num_machines, order_pipeline, max_batch_size=batch_size, mttf=mttf, a=a,
                    ordering_cost=co, emergency_cost=ce, bsp=True, bsp_n=n, gamma=0.99, sorted_degradation=True)
        pol = PPO(MaskableActorCriticPolicy, env=env, verbose=0)
        pol.learn(800000)
    elif policy == 'ppo_probsp':
        n, xo = find_parameters(experiment_data)
        env = InventoryRS(num_machines, order_pipeline, max_batch_size=batch_size, mttf=mttf, a=a, gamma=0.99,
                          ordering_cost=co, emergency_cost=ce, bsp=False, probsp=True, probsp_n=n, probsp_xo=xo,
                          sorted_degradation=True)
        pol = PPO(MaskableActorCriticPolicy, env=env, verbose=0)
        pol.learn(800000)
    elif policy == 'dqn_bsp':
        n, xo = find_parameters(experiment_data)
        env = InventoryRS(num_machines, order_pipeline, max_batch_size=batch_size, mttf=mttf, a=a,
                          ordering_cost=co, emergency_cost=ce, bsp=True, bsp_n=n, gamma=0.99, sorted_degradation=True)
        pol = DDQN(env, train_every=2, replay_memory_size=400000, update_target_every=10000)
        pol.learn(800000)
    elif policy == 'dqn_probsp':
        n, xo = find_parameters(experiment_data)
        env = InventoryRS(num_machines, order_pipeline, max_batch_size=batch_size, mttf=mttf, a=a, gamma=0.99,
                          ordering_cost=co, emergency_cost=ce, bsp=False, probsp=True, probsp_n=n, probsp_xo=xo,
                          sorted_degradation=True)
        pol = DDQN(env, train_every=2, replay_memory_size=400000, update_target_every=10000)
        pol.learn(800000)
    else:
        return 0

    avg_cost, std_cost, avg_FR, std_FR, avg_ES, std_ES = evaluate_policy(env, pol, processors=4)
    data = [policy,n,xo,str(env), env.num_machines, env.order_pipeline.p, env.mttf, env.degradation_a, env._ordering_cost,
            env._emergency_cost, env.max_batch_size, env.sorted_degradation,
            avg_cost, std_cost, avg_FR, std_FR, avg_ES, std_ES]
    with open("results.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)
    file.close()

if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor(1) as executor:
        for r in executor.map(run_experiment, combinations):
            print(r)