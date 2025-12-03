import csv
import math
import pprint
import random
import time
import warnings
from collections import namedtuple, deque
import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


from NewEnvironment import Inventory

torch.set_default_device("cpu")

warnings.filterwarnings(action='ignore')

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'mask'))

PrioritizedTransition = namedtuple('Transition',
                                   ('state', 'action', 'next_state', 'reward', 'mask', 'weight'))


class FastReplayMemory:
    """
    Fast replay memory using pre-allocated numpy arrays.
    ~10-20x faster than deque-based implementation.
    """

    def __init__(self, capacity, state_dim, action_dim, device='cpu'):
        """
        Args:
            capacity: Maximum number of transitions to store
            state_dim: Dimension of state space
            action_dim: Dimension of action space (for masks)
            device: torch device for batch tensors
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # Pre-allocate arrays (much faster than deque)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.masks = np.zeros((capacity, action_dim), dtype=bool)

        self.position = 0
        self.size = 0

    def push(self, state, action, next_state, reward, mask):
        """Save a transition - optimized for minimal overhead"""
        # Convert tensors to numpy efficiently
        if isinstance(state, torch.Tensor):
            self.states[self.position] = state.squeeze().cpu().numpy()
        else:
            self.states[self.position] = state

        if isinstance(action, torch.Tensor):
            self.actions[self.position] = action.item()
        else:
            self.actions[self.position] = action

        if isinstance(next_state, torch.Tensor):
            self.next_states[self.position] = next_state.squeeze().cpu().numpy()
        else:
            self.next_states[self.position] = next_state

        if isinstance(reward, torch.Tensor):
            self.rewards[self.position] = reward.item()
        else:
            self.rewards[self.position] = reward

        if isinstance(mask, torch.Tensor):
            self.masks[self.position] = mask.squeeze().cpu().numpy()
        else:
            self.masks[self.position] = mask

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """
        Fast batch sampling - returns tensors directly on target device.
        ~10x faster than sampling deque and converting.
        """
        # Fast numpy random sampling
        indices = np.random.randint(0, self.size, size=batch_size)

        # Single tensor conversion (much faster than individual conversions)
        batch_states = torch.from_numpy(self.states[indices]).to(self.device)
        batch_actions = torch.from_numpy(self.actions[indices]).to(self.device)
        batch_next_states = torch.from_numpy(self.next_states[indices]).to(self.device)
        batch_rewards = torch.from_numpy(self.rewards[indices]).to(self.device)
        batch_masks = torch.from_numpy(self.masks[indices]).to(self.device)

        return batch_states, batch_actions, batch_next_states, batch_rewards, batch_masks

    def __len__(self):
        return self.size

    def reset(self):
        """Clear the buffer"""
        self.position = 0
        self.size = 0


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        # TODO: Very time consuming
        return random.sample(self.memory, batch_size)

    def n_step_sample(self, batch_size, n_step):
        start_idx = [random.randint(0, len(self) - n_step) for _ in range(batch_size)]

        _temp_memory = list(self.memory)
        for n in range(n_step):
            _temp = []
            for idx in start_idx:
                _temp.append(_temp_memory[idx + n])
            yield tuple(_temp)

    def __len__(self):
        return len(self.memory)

    def reset(self):
        self.memory.clear()


class PrioritizedExperienceReplay:

    def __init__(self, capacity):
        self._capacity = capacity
        self.memory = deque([], maxlen=capacity)
        # TODO: Change the below 2 to numpy to increase speed, keep only transition in a queue
        self.priorities = deque([], maxlen=capacity)
        self.indices = deque([], maxlen=capacity)
        self.full = False
        self._alpha = 0.7  # How much prioritization is used when sampling, higher more prioritized
        self._beta = 0.1  # Correcting the bias due to prioritization, it is annealed to 1 during the training
        self._maximum_weight = 1

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
        if not self.full and len(self) == self._capacity:
            self.indices.append(len(self) - 1)
            self.full = True
            self.priorities.append(self._maximum_weight)
        if not self.full:
            self.priorities.append(self._maximum_weight)
            self.indices.append(len(self) - 1)

    def sample(self, batch_size):
        """
        Sample from PER and return the indices, the transition, and the normalized Importance Sampling weights
        :param batch_size: Batch Size
        :return: list of Indices, Transitions, List of weights
        """
        # Update the weights first
        self._update_weights()

        # sample indices, then transitions
        indices = random.choices(population=self.indices,
                                 weights=self.priorities,
                                 k=batch_size)
        transition_batch = [self.memory[i] for i in indices]

        # Compute importance sampling weights
        is_weights = [(len(self) * self.priorities[idx]) ** (-self._beta) for idx in indices]
        max_is_weight = max(is_weights)
        is_weights = [x / max_is_weight for x in is_weights]

        # Increment Beta with a cap of 1
        self._beta = min(1., self._beta + 0.00001)
        return indices, transition_batch, is_weights

    def __len__(self):
        if self.full:
            return self._capacity
        else:
            return len(self.memory)

    def _update_weights(self):
        _sum_priorities = sum(self.priorities)
        _priority_list = [priority ** self._alpha / _sum_priorities for priority in self.priorities]
        self.priorities = deque(_priority_list)

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = min(abs(priority.item()), self._maximum_weight)




class DQN(nn.Module):

    def __init__(self, env: Inventory):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(len(env.reset()[0]), 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 128)
        self.layer4 = nn.Linear(128, env.action_space.n)
        with torch.no_grad():
            self.layer4.bias.fill_(0)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)


class MaskedDQN:

    def __init__(self,
                 env: Inventory,
                 b_size=64,
                 nn_training_epochs=1,
                 gamma=0.99,
                 eps_start=0.99,
                 eps_end=0.01,
                 eps_decay=90000,
                 tau=0.9,
                 learning_rate=1e-4,
                 replay_memory_size: int = 600000,
                 train_every: int = 2,
                 update_target_every: int = 20000,
                 evaluate_every: int = 0,
                 verbose=0,
                 write_tensorboard:bool=False,
                 log_dir="runs/DQN/",
                 ):
        self.env = env
        self._b_size = b_size
        self.nn_training_epochs = nn_training_epochs
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_end = eps_end
        self.eps = eps_start
        self.tau = tau
        self.learning_rate = learning_rate
        self.train_every = train_every
        self.update_target_every = update_target_every
        self._replay_memory_size = replay_memory_size
        self.evaluate_every = evaluate_every
        self.verbose = verbose
        self.dqn_model = DQN

        self.env = env
        self._b_size = b_size
        self.device = torch.device("cpu")

        # Get dimensions from environment
        obs, _ = env.reset()
        state_dim = len(obs)
        action_dim = env.action_space.n

        # NEW: Use fast replay memory
        self.replay_memory = FastReplayMemory(
            capacity=replay_memory_size,
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.device)
        # Inner data
        # self.replay_memory = ReplayMemory(self._replay_memory_size)
        self.policy_net = self.dqn_model(self.env).to(self.device)
        self.target_net = self.dqn_model(self.env).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True)

        # Learning Data
        self.training_steps = 0
        self.loss = 0

        self.actions_tensor = torch.arange(start=0, step=1, end=self.env.action_space.n+1, dtype=torch.int64)

        if write_tensorboard:
            log_dir = log_dir + time.strftime("%Y%m%d-%H%M%S")
            self.writer = SummaryWriter(log_dir)
            self.write_tensorboard = True
        else:
            self.write_tensorboard = False

    def save(self, path: str):
        torch.save(self.policy_net, path)

    def load(self, path: str):
        self.policy_net = torch.load(path, weights_only=False)
        self.target_net = torch.load(path, weights_only=False)

    def learn(self, training_steps: int = 200000, seed: int = None):
        print(f"Starting Learning for {training_steps:,d} steps")
        action_buffer = deque(maxlen=500)
        learn_start_time = time.time()
        _env = self.env.copy()
        obs, info = _env.reset(seed=seed)
        state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        for t in tqdm.tqdm(range(training_steps)):
            action, _ = self.predict(state, _env.action_masks())
            obs, reward, terminated, truncated, info = _env.step(action.item(), verbose=False)
            if self.write_tensorboard:
                action_buffer.append(action)
                if t % 20000 == 0:
                    self.writer.add_histogram("env/actions", np.array(action_buffer), t)
                self.writer.add_scalar("learn/avg costs", float(info['average_cost']), self.training_steps)
                self.writer.add_scalar("learn/E[S].", float(info['average_inventory']), self.training_steps)
                self.writer.add_scalar("learn/FR", float(info['fill_rate']), self.training_steps)
                self.writer.add_scalar("Costs/step", info['step_cost'], t)
                self.writer.add_scalar("Costs/avg", info["average_cost"], t)
                self.writer.add_scalar("Costs/holding", info['holding_costs'], t)
                self.writer.add_scalar("Costs/ordering", info["ordering_costs"], t)
                self.writer.add_scalar("Costs/emergency", info["emergency_costs"], t)
            reward = torch.tensor([reward], dtype=torch.float32).to(self.device)
            mask = torch.tensor(_env.action_masks()).unsqueeze(0).to(self.device)
            next_state = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            self.replay_memory.push(state, action, next_state, reward, mask)

            # Move to the next state
            state = next_state

            if t > 0 and t % self.train_every == 0:
                for _ in range(self.nn_training_epochs):
                    self._optimize_model()

            self._soft_update(t)

            if t > 0 and self.evaluate_every > 0 and t % self.evaluate_every == 0:
                self._evaluate_model()

            if self.write_tensorboard:
                self.writer.add_scalar("eps", self.eps, self.training_steps)

            self._print_info(t, info)

            self.training_steps = t
        end_learn_time = time.time()
        print(f"Finished Learning in {end_learn_time - learn_start_time:.2f}s")

    def predict(self, observation, action_masks, deterministic=False,):
        """
        This method finds action either using the policy with an increasing probability, or chooses a random
         action with decreasing probability.
        :param observation:
        :param action_masks:
        :param deterministic:
        :return:
        """
        mask = action_masks
        mask_tensor = torch.tensor(mask).unsqueeze(0).to(self.device)
        if deterministic:
            if type(observation) is np.ndarray:
                observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
            q_values = self.policy_net(observation)
            masked_q_values = torch.masked_fill(q_values, ~mask_tensor, -1000)
            index = masked_q_values.argmax(dim=1).to(torch.long)
            # TODO: Unsequeeze in place of the list use
            # to_return = torch.tensor([[int(index)]], dtype=torch.int)
            return index, None
        sample = random.random()
        self.update_eps()
        if self.write_tensorboard:
            self.writer.add_scalar("eps", self.eps, self.training_steps)
        self.training_steps += 1
        if sample > self.eps:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                q_values = self.policy_net(observation)
                masked_q_values = torch.masked_fill(q_values, ~mask_tensor, -1000)
                index = masked_q_values.argmax(dim=1).to(torch.long).to(self.device)
                # to_return = torch.tensor([[int(index)]], dtype=torch.long)
                return index, None
        else:
            mask = mask.astype(dtype=np.int8)
            # mask = tuple(mask)
            return torch.tensor([self.env.action_space.sample(mask=mask)], dtype=torch.long,
                                device=self.device), None

    def update_eps(self):
        self.eps = self.eps_end + (self.eps_start - self.eps_end) * \
                   math.exp(-1. * self.training_steps / self.eps_decay)

    def _get_next_value(self, non_final_next_states, mask_batch):
        target_q_values = self.target_net(non_final_next_states)
        masked_target_q_value = torch.masked.masked_tensor(target_q_values, mask_batch).to(self.device)
        _values = masked_target_q_value.amax(1)
        _actions_prime = masked_target_q_value.argmax(1)
        return torch.tensor(_values.get_data()), torch.tensor(_actions_prime.get_data())

    def _optimize_model(self):
        if len(self.replay_memory) < self._b_size:
            return

        # With Fast Replay Memory:
        state_batch, action_batch, next_state, reward_batch, mask_batch = \
            self.replay_memory.sample(self._b_size)

        # Action batch needs unsqueeze for gather
        action_batch = action_batch.unsqueeze(-1)

        # For regular (old) replay memory
        # transitions = self.replay_memory.sample(self._b_size)
        # # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # # detailed explanation). This converts batch-array of Transitions
        # # to Transition of batch-arrays.
        # batch = Transition(*zip(*transitions))
        #
        # next_state = torch.cat(batch.next_state)
        # state_batch = torch.cat(batch.state)
        # action_batch = torch.cat(batch.action)
        # action_batch = action_batch.unsqueeze(-1)
        # reward_batch = torch.cat(batch.reward)
        # mask_batch = torch.cat(batch.mask)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self._b_size, device=self.device)
        with torch.no_grad():
            next_state_values, next_actions = self._get_next_value(next_state, mask_batch)
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch.unsqueeze(1)
        assert state_action_values.shape == expected_state_action_values.shape, (f"Shape mismatch between state and "
                                                                                 f"expectation")
        if self.write_tensorboard:
            td_error = torch.mean(torch.abs(state_action_values - expected_state_action_values))
            self.writer.add_scalar("model/td-error", td_error, self.training_steps)

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        self.loss = criterion(state_action_values, expected_state_action_values)
        if self.write_tensorboard:
            self.writer.add_scalar("model/loss", self.loss.item(), self.training_steps)

        self.optimizer.zero_grad()
        self.loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def _print_info(self, t, info):
        if self.verbose == 1 and t % 10000 == 0 and t > 0:
            print(f"---Train Data-----\n"
                  f"| t={t}          \n"
                  f"| eps={self.eps:.2f}  \n"
                  f"| loss={self.loss}\n"
                  f"| Avg Cost={float(info['Average Cost']):.3f}\n"
                  f"| PM={float(info['Preventive Maintenance']):.3f}\n"
                  f"| CM={float(info['Corrective Maintenance']):.3f}\n"
                  f"| Lost Sales={float(info['Lost Sales']):.3f}\n"
                  f"------------------")

    def _soft_update(self, t):
        if t > 0 and t % self.update_target_every == 0:
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (
                        1 - self.tau)
            self.target_net.load_state_dict(target_net_state_dict)

    def _evaluate_model(self):
        _env = self.env.copy()
        state, info = _env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        for t in range(10000):
            # Initialize the environment and get its state
            action, _ = self.predict(state, action_masks=_env.action_masks(), deterministic=True)
            observation, reward, terminated, truncated, info = _env.step(action.item())
            state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        if self.write_tensorboard:
            self.writer.add_scalar("eval/avg costs", float(info['average_cost']), self.training_steps)
            self.writer.add_scalar("eval/E[S].", float(info['average_inventory']), self.training_steps)
            self.writer.add_scalar("eval/FR", float(info['fill_rate']), self.training_steps)
            self.writer.add_scalar("eval/step", info['step_cost'], t)
            self.writer.add_scalar("eval/avg", info["average_cost"], t)
            self.writer.add_scalar("eval/holding", info['holding_costs'], t)
            self.writer.add_scalar("eval/ordering", info["ordering_costs"], t)
            self.writer.add_scalar("eval/emergency", info["emergency_costs"], t)


class MaskedDoubleDQN(MaskedDQN):

    def __init__(self,
                 env: Inventory,
                 b_size=64,
                 nn_training_epochs=1,
                 gamma=0.99,
                 eps_start=0.99,
                 eps_end=0.01,
                 eps_decay=90000,
                 tau=0.9,
                 learning_rate=1e-4,
                 replay_memory_size: int = 600000,
                 train_every: int = 2,
                 update_target_every: int = 20000,
                 evaluate_every: int = 0,
                 verbose=0,
                 write_tensorboard: bool = False,
                 log_dir="runs/DQN/",
                 ):
        super().__init__(env, b_size, nn_training_epochs, gamma, eps_start, eps_end, eps_decay, tau, learning_rate,
                         replay_memory_size, train_every, update_target_every, evaluate_every, verbose,
                         write_tensorboard=write_tensorboard, log_dir=log_dir)

    def _get_next_value(self, non_final_next_states, mask_batch):
        """ Compute the error in TD using the target action, and the policy q value:
        Error = R_{t+1} + gamma Q_{target}(S_{t+1}, best action from policy network (S_{t+1}))
        """
        policy_q_values = self.policy_net(non_final_next_states)
        policy_actions = torch.masked_fill(policy_q_values, ~mask_batch, -1000).argmax(dim=1)
        target_q_values = self.target_net(non_final_next_states)
        _values = target_q_values.gather(1, policy_actions.unsqueeze(1))
        # _actions_prime = target_q_values.argmax(1)
        # _actions_prime = torch.masked_fill(policy_q_values, ~mask_batch, -1000).argmax(dim=1)
        return _values, policy_actions.flatten()

