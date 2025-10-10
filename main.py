from Auxiliaries import evaluate_policy
from Policies import ProBSP
from NewEnvironment import Inventory, GeometricOrderPipeline, InventoryRS
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib import MaskablePPO as PPO
from MaskedDQN import MaskedDoubleDQN as DDQN


"""
This file provides an example on defining an environment and learning a DQN, and PPO policies.
The inventory problem has the following parameters:
M=2, p=0.33, B=3, MTTF=10, Co=2, Ce=5
"""

num_machines = 2
lead_times_p = 1/3
max_batch_size = 3
mttf = 10
sort_degradation = True  # Sort degradation to benefit from the reduction in state space

order_pipeline = GeometricOrderPipeline(num_machines, lead_times_p)
inventory = Inventory(machines=num_machines,
                      order_pipeline=order_pipeline,
                      mttf=mttf,
                      sorted_degradation=sort_degradation)

n, xo = 0, 60
probsp = ProBSP(env=inventory, n=n, xo=xo, max_batch_size=max_batch_size)

evaluate_policy(env=inventory, policy=probsp, replication=8, processors=4)

# Let us learn using a PPO policy
print("Learning a policy using PPO")
ppo = PPO(MaskableActorCriticPolicy, env=inventory, verbose=0)
ppo.learn(200000)
print("Finished Learning PPO policy, evaluating .......")
evaluate_policy(env=inventory, policy=ppo, replication=8, processors=4)

# Let us learn using a DQN policy
print("Learning a policy using DQN")
dqn = DDQN(env=inventory, verbose=0, eps_decay=20000)
dqn.learn(200000)
print("Finished Learning DQN policy, evaluating .......")
evaluate_policy(env=inventory, policy=dqn, replication=8, processors=4)


# When reward shaping is included using the ProBSP
inventory_rs = InventoryRS(machines=num_machines,
                           order_pipeline=order_pipeline,
		                   mttf=mttf,
		                   sorted_degradation=sort_degradation,
                           probsp=True,
                           bsp=False,
                           probsp_xo=xo,
                           probsp_n=n,
                           gamma=0.99
                           )

# Again, let us learn using a PPO policy
print("Learning a policy using PPO")
ppo = PPO(MaskableActorCriticPolicy, env=inventory_rs, verbose=0)
ppo.learn(200000)
print("Finished Learning PPO policy, evaluating .......")
evaluate_policy(env=inventory_rs, policy=ppo, replication=8, processors=4)

# Let us learn using a DQN policy
print("Learning a policy using DQN")
dqn = DDQN(env=inventory_rs, verbose=0, eps_decay=20000)
dqn.learn(200000)
print("Finished Learning DQN policy, evaluating .......")
evaluate_policy(env=inventory_rs, policy=dqn, replication=8, processors=4)

