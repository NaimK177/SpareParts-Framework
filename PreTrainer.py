import math
from typing import Union

import torch
import torch.nn as nn
from Policies import ProBSP, BaseStockPolicy
from NewEnvironment import Inventory
from sb3_contrib import MaskablePPO as PPO
from MaskedDQN2 import MaskedDoubleDQN
import numpy as np
import gymnasium as gym


# --- 0. HELPER FUNCTION: Soft Targets ---
def create_target_distribution(heuristic_action, action_mask, num_actions, high_prob=0.8):
	"""
	Creates a soft target distribution based on the heuristic action and mask.
	Assigns 'high_prob' to the expert choice and distributes the rest among other valid actions.
	"""

	heuristic_action = int(heuristic_action)
	target_dist = torch.zeros(num_actions, dtype=torch.float32)

	# Identify valid indices
	valid_indices = torch.nonzero(torch.tensor(action_mask)).flatten()

	# Assign High Probability to Expert Choice
	target_dist[heuristic_action] = high_prob

	# Distribute remaining probability
	remaining_prob = 1.0 - high_prob
	num_other_valid = action_mask.sum() - 1

	if num_other_valid > 0:
		share_per_action = remaining_prob / num_other_valid
		for idx in valid_indices:
			if idx != heuristic_action:
				target_dist[idx] = share_per_action
	else:
		# If only 1 action is valid, it gets 100%
		target_dist[heuristic_action] = 1.0

	return target_dist


device = torch.device("cpu")

def pre_train_ppo(model: Union[PPO], inventory:Inventory, teacher: Union[ProBSP, BaseStockPolicy],
                  n_steps:int=20000, gae_steps:int=500, gamma:float=0.99, b_size:int=32, epochs:int=100):
	observations = []
	target_distributions = []  # Changed from expert_actions to distributions
	rewards_stream = []

	obs, _ = inventory.reset()

	for _ in range(n_steps + gae_steps):
		# Get Heuristic Action
		action, _ = teacher.predict(obs)

		# Get Mask (Crucial for Soft Targets)
		# Ensure your env has this method, or use ppo.env.action_masks() if using VecEnv
		current_mask = inventory.action_masks()

		# Store data only for training steps
		if len(observations) < n_steps:
			observations.append(obs)

			# GENERATE SOFT TARGET
			soft_target = create_target_distribution(
				heuristic_action=action,
				action_mask=current_mask,
				num_actions=inventory.action_space.n,
				high_prob=0.8)
			target_distributions.append(soft_target.numpy())

		obs, reward, terminated, truncated, _ = inventory.step(action)
		rewards_stream.append(reward)

		if terminated or truncated:
			obs, _ = inventory.reset()

	# --- Calculate Returns ---
	calculated_returns = []
	# Optimized linear calculation
	for t in range(n_steps):
		G_t = 0
		discount = 1.0
		limit = min(t + gae_steps, len(rewards_stream))
		for k in range(t, limit):
			G_t += discount * rewards_stream[k]
			discount *= gamma
		calculated_returns.append(G_t)

	# --- PREPARE TENSORS ---
	obs_tensor = torch.tensor(np.array(observations), dtype=torch.float32).to(device)
	ret_tensor = torch.tensor(np.array(calculated_returns), dtype=torch.float32).view(-1, 1).to(device)

	# Shape: (N, Num_Actions) -> Probability Distributions
	act_tensor = torch.tensor(np.array(target_distributions), dtype=torch.float32).to(device)
	# Use KL Divergence for distributions
	actor_criterion = nn.KLDivLoss(reduction='batchmean', log_target=True)

	critic_criterion = nn.MSELoss()

	# --- 3. IN-PLACE TRAINING ---
	# print(f"Pre-training PPO Policy AND Value Network on {device}...")

	policy_network = model.policy
	optimizer = torch.optim.Adam(policy_network.parameters(), lr=4e-4)
	policy_network.train()

	data_len = len(obs_tensor)

	def get_dummy_masks(batch_size, env):
		# For MaskablePPO forward pass, we allow all actions during pre-training
		# (The KL loss naturally handles invalid actions via the soft target having 0.0 prob)
		if isinstance(env.action_space, gym.spaces.Discrete):
			n_actions = env.action_space.n
			return torch.ones((batch_size, n_actions), dtype=torch.bool).to(device)
		return None

	for epoch in range(epochs):
		indices = torch.randperm(data_len)
		total_loss = 0

		for i in range(0, data_len, b_size):
			idx = indices[i: i + b_size]

			batch_obs = obs_tensor[idx]
			batch_targets = act_tensor[idx]  # Soft targets
			batch_rets = ret_tensor[idx]

			# 1. Critic Forward Pass
			value_output = policy_network.predict_values(batch_obs)

			# 2. Actor Forward Pass
			if isinstance(model, PPO):
				dummy_masks = get_dummy_masks(len(batch_obs), inventory)
				dist = policy_network.get_distribution(batch_obs, dummy_masks)
			else:
				dist = policy_network.get_distribution(batch_obs)

			# 3. Loss Calculation
			critic_loss = critic_criterion(value_output, batch_rets)

			# Get log probabilities for KL divergence
			logits = dist.distribution.logits
			# log_probs = F.log_softmax(logits, dim=-1)

			# Ensure targets are probabilities (should sum to 1)
			# If using KL divergence, target should also be in log space
			target_log_probs = torch.log(batch_targets.clamp(min=1e-8))

			actor_loss = actor_criterion(logits, target_log_probs)

			# Combined Loss
			loss = actor_loss + 0.5 * critic_loss

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			total_loss += loss.item()
	policy_network.train(False)
	print("Pre-training complete. Weights initialized")
	return model


class MaskedDDQNPro(MaskedDoubleDQN):

	def __init__(self,
	             env: Inventory,
	             probsp: ProBSP,
	             nn_training_epochs=1,
	             b_size=64,
	             gamma=0.99,
	             eps_start=0.99,
	             eps_end=0.01,
	             eps_decay=90000,
	             tau=0.9,
	             learning_rate=1e-4,
	             replay_memory_size: int = 300000,
	             train_every: int = 2,
	             update_target_every: int = 10000,
	             evaluate_every: int = 0,
	             verbose=0,
	             write_tensorboard=False,
	             log_dir="runs/DDQN/",
	             ):
		super().__init__(env, b_size, nn_training_epochs, gamma, eps_start, eps_end, eps_decay, tau, learning_rate,
		                 replay_memory_size,
		                 train_every, update_target_every, evaluate_every, verbose, write_tensorboard=write_tensorboard,
		                 log_dir=log_dir)
		self.probsp = probsp
		self.probsp_period = 40000
		self.current_period = 1

	def predict(self, observation, action_masks, deterministic=False):
		if self.current_period < self.probsp_period and not deterministic:
			action, _ = self.probsp.predict(obs=observation.flatten().numpy())
			return torch.tensor([action], dtype=torch.long), None
		else:
			return super().predict(observation, action_masks, deterministic)

	def _optimize_model(self):
		self.current_period += 1
		if self.current_period > self.probsp_period + self._b_size:
			super()._optimize_model()

	def _print_info(self, t, info):
		if self.verbose == 1 and t % 10000 == 0 and t > 0:
			print(f"---Train Data-----\n"
			      f"| t={t}          \n"
			      f"| eps={self.eps:.2f}  \n"
			      f"| loss={self.loss}\n"
			      f"| Avg Cost={float(info['average_cost']):.3f}\n"
			      f"------------------")

	def update_eps(self):
		self.eps = self.eps_end + (self.eps_start - self.eps_end) * \
		           math.exp(-1. * (self.training_steps - self.probsp_period) / self.eps_decay)