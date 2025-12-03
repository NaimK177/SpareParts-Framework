from Auxiliaries import evaluate_policy
from Policies import ProBSP
from NewEnvironment import Inventory, GeometricOrderPipeline, InventoryRS
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib import MaskablePPO as PPO
from PreTrainer import pre_train_ppo, MaskedDDQNPro


num_machines = 30
lead_times_p = 0.2
max_batch_size = 5
mttf = 10
sort_degradation = True
n, xo = 6, 56.25
GAMMA = 0.99

order_pipeline = GeometricOrderPipeline(num_machines, lead_times_p)
inventory = Inventory(machines=num_machines,
					  order_pipeline=order_pipeline,
					  mttf=mttf,
					  sorted_degradation=sort_degradation)
inventory_rs = InventoryRS(
	machines=num_machines,
	order_pipeline=order_pipeline,
	mttf=mttf,
	sorted_degradation=sort_degradation,
	bsp=False,
	probsp=True,
	probsp_n=n,
	probsp_xo=xo,
	gamma=GAMMA
)
# Initialize Heuristic
probsp = ProBSP(env=inventory, n=n, xo=xo, max_batch_size=max_batch_size)

# print("Evaluating ProBSP")
# evaluate_policy(inventory, policy=probsp)

# Initialize PPO on the correct device
ppo_pi = PPO(MaskableActorCriticPolicy, env=inventory, verbose=0, gamma=GAMMA)
ppo_rs = PPO(MaskableActorCriticPolicy, env=inventory_rs, verbose=0, gamma=GAMMA)
#
# print("Training PPO using RS for 800000")
# ppo_rs.learn(800000)
# print("Evaluating PPO-RS")
# evaluate_policy(inventory, policy=ppo_rs)

# print("Pre-Training PPO")
# ppo_pi = pre_train_ppo(ppo_pi, inventory,probsp, b_size=32, epochs=200)
# print("Evaluating pre-trained ppo")
# evaluate_policy(inventory, policy=ppo_pi)
#
# print("Retraining PPO for 800000")
# ppo_pi.learn(800000)
# print("Evaluating post-trained ppo")
# evaluate_policy(inventory, policy=ppo_pi)

ddqn_pro = MaskedDDQNPro(inventory, probsp, verbose=1, write_tensorboard=True)
ddqn_pro.learn(800000)

evaluate_policy(inventory, ddqn_pro)

