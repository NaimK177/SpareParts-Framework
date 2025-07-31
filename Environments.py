"""
Environment Module for Spare Parts Management using Gymnasium

This module implements custom environments for spare parts inventory management.
The environment simulates the dynamics of spare parts inventory control in a maintenance system.
It is based on the gymnasium environment.

The environment implements the standard Gymnasium methods:
- reset(): Resets the environment to initial state
- step(action): Executes one timestep within the environment
"""

import random
from typing import Tuple

import gym as gym
import numpy as np
from gym import spaces
from gym.core import ActType, ObsType
from gym.utils import seeding


MAX_BATCH_SIZE = 3
# seed = 123



class DeterministicOrder:

    def __init__(self, lead_time: float, batch_ordering: bool = False):
        self.lead_time = int(lead_time)
        self.arrivals = np.array([0 for _ in range(self.lead_time)], dtype=int)
        self.batch_ordering = batch_ordering
        self.partial_expedition = True

    def __repr__(self):
        return f"Deterministic Orders pipeline with vector={self.arrivals}"

    def __len__(self):
        return np.sum(self.arrivals)

    def add_order(self, a):
        self.arrivals[-1] += a

    def get_arrivals(self):
        arrivals = self.arrivals[0]
        _temp = np.zeros_like(self.arrivals)
        for i in range(len(self.arrivals) - 1):
            _temp[i] = self.arrivals[i + 1]
            _temp[-1] = 0
        self.arrivals = _temp
        return arrivals

    def expedite_arrivals(self, expedites_needed):
        if not self.partial_expedition and self.batch_ordering:
            arrivals = 0
            for i in range(self.lead_time):
                if self.arrivals[i] > 0 and expedites_needed > 0:
                    arrivals += self.arrivals[i]
                    expedites_needed -= self.arrivals[i]
                    self.arrivals[i] = 0
                    if expedites_needed <= 0:
                        return arrivals
            return arrivals
        else:
            arrivals = 0
            for i in range(self.lead_time):
                if self.arrivals[i] > 0 and expedites_needed > 0:
                    for j in range(self.arrivals[i]):
                        arrivals += 1
                        self.arrivals[i] -= 1
                        expedites_needed -= 1
                        if expedites_needed == 0:
                            break
            return arrivals

    def reset(self):
        self.arrivals = np.array([0 for _ in range(self.lead_time)], dtype=int)

    def get_state(self):
        _arr = np.zeros(MAX_BATCH_SIZE)
        for i in range(MAX_BATCH_SIZE):
            _arr[i] = np.sum(self.arrivals == i+1)
        return _arr


class StochasticBatchOrder:

    def __init__(self, p: float = 0.1, max_orders: int = 1):
        self.p = p
        self.max_orders = max_orders
        self.arrivals = np.zeros(max_orders, dtype=int)
        self.partial_expedition = True

    def __repr__(self):
        return f"Stochastic Orders pipeline with p={self.p}"

    def __len__(self):
        return np.sum(self.arrivals)

    def add_order(self, a):
        idx = np.where(self.arrivals == 0)[0][0]
        self.arrivals[idx] += a

    def get_arrivals(self):
        arrivals_bool = np.random.binomial(n=1, p=self.p, size=self.max_orders)
        if np.sum(arrivals_bool * self.arrivals) == 0:
            return 0
        arrivals_idx = arrivals_bool.nonzero()
        arrivals = np.sum(arrivals_bool * self.arrivals)
        self.arrivals[arrivals_idx] = 0
        return arrivals

    def expedite_arrivals(self, expedites_needed) -> int:
        if self.partial_expedition:
            # TODO: check
            arrivals = 0
            while arrivals < expedites_needed and np.sum(self.arrivals) > 0:
                orders_idx = np.where(self.arrivals > 0)[0]
                sample_batch = np.random.choice(orders_idx, size=1)
                arrivals += 1
                self.arrivals[sample_batch] -= 1
            return arrivals
        else:
            arrivals = 0
            for i in range(self.max_orders):
                if self.arrivals[i] > 0 and expedites_needed > 0:
                    arrivals += self.arrivals[i]
                    expedites_needed -= self.arrivals[i]
                    self.arrivals[i] = 0
                    if expedites_needed <= 0:
                        return arrivals
            return 0

    def reset(self):
        self.arrivals = np.array([0 for _ in range(self.max_orders)], dtype=int)

    def get_state(self):
        _arr = np.zeros(MAX_BATCH_SIZE)
        for i in range(MAX_BATCH_SIZE):
            _arr[i] = np.sum(self.arrivals == i+1)
        return _arr


class Inventory(gym.Env):

    def __init__(self,
                 machines: int = 1,
                 lead_time_p: float = 1,
                 mttf: float = 10.,
                 a: float = 1.,
                 ordering_cost: float = 2,
                 emergency_cost: float = 5,
                 deterministic_lead_time: bool = False,
                 sorted_degradation: bool = False):
        """
        Inventory model with multiple machines degrading following a gamma process.
        The inventory model takes number of spare parts as decision, and spare parts are ordered.
        They can arrive in the future following some lead time distribution, or deterministically.

        When a maintenance is due without having spare parts in inventory, emergency cost incurs to expedite an
        existing order.
        :param machines:
        :param mean_lead_time:
        :param mttf:
        :param a:
        :param ordering_cost:
        :param emergency_cost:
        :param deterministic_lead_time:
        :param batch_ordering:
        :param sorted_degradation:
        """
        self.lead_time_p = lead_time_p
        self.bino_prob = lead_time_p
        self.inventory_capacity = machines
        self._maintenance_threshold = 100
        self.num_machines = machines
        self.mean_lead_time = 1 / lead_time_p
        self.mttf = mttf
        self.degradation_a = a
        self.degradation_u = (mttf * a - 0.5) / 100
        self.deterministic_lead_time = deterministic_lead_time
        self.batch_ordering = True
        if self.batch_ordering:
            self.orders_pipeline = StochasticBatchOrder(p=self.bino_prob, max_orders=self.inventory_capacity)
        if deterministic_lead_time:
            if self.batch_ordering:
                self.orders_pipeline = DeterministicOrder(self.mean_lead_time, self.batch_ordering)
            else:
                self.orders_pipeline = DeterministicOrder(self.mean_lead_time)
        self.sorted_degradation = sorted_degradation

        self.degradations = np.zeros((self.num_machines,))
        self.inventory_level = 0
        self.outstanding_orders = 0

        self._holding_cost = 1.
        self._ordering_cost = ordering_cost
        self._emergency_cost = emergency_cost

        self.max_cost = self._emergency_cost * self.num_machines

        # Total Costs
        # Computed Cumulative Costs
        self._holding_total = 0.
        self._ordering_total = 0.
        self._emergency_total = 0.
        self._total_cost = 0

        self.total_expedited_orders = 0
        self.total_maintenance = 1

        self.time_step = 1
        self._average_stock = self.inventory_level

        # Observation and Action space
        self.action_space = spaces.Discrete(self.inventory_capacity + 1)
        self._action_array = np.asarray([a for a in range(machines + 1)])
        deg_low = np.array([0.] * self.num_machines)
        deg_high = np.array([1.] * self.num_machines)
        inventory_low = np.array([0.])
        inventory_high = np.array([1.])
        outstanding_low = np.array(MAX_BATCH_SIZE * [0.])
        outstanding_high = np.array(MAX_BATCH_SIZE * [1.])
        np_low_values = np.concatenate(
            [
                deg_low,
                inventory_low,
                outstanding_low
            ]
        ).astype(np.float32)
        np_high_values = np.concatenate(
            [deg_high,
             inventory_high,
             outstanding_high
             ]
        ).astype(np.float32)
        self.observation_space = spaces.Box(low=np_low_values,
                                            high=np_high_values,
                                            dtype=np.float32)

        self.parts_array = [x + 1 for x in range(MAX_BATCH_SIZE)]

    def __eq__(self, other):
        checks = [self.bino_prob == other.bino_prob,
                  self.num_machines == other.num_machines,
                  self._maintenance_threshold == other._maintenance_threshold,
                  self.mean_lead_time == other.mean_lead_time,
                  self.mttf == other.mttf,
                  self.degradation_a == other.degradation_a,
                  self._holding_cost == other._holding_cost,
                  self._ordering_cost == other._ordering_cost,
                  self._emergency_cost == other._emergency_cost]
        for check in checks:
            if check:
                pass
            else:
                return False
        return True

    def __hash__(self):
        return hash(str(self))

    def copy(self):
        dummy = Inventory(machines=self.num_machines, lead_time_p=self.lead_time_p,
                          mttf=self.mttf, a=self.degradation_a,
                          ordering_cost=self._ordering_cost, emergency_cost=self._emergency_cost,
                          deterministic_lead_time=self.deterministic_lead_time)
        return dummy

    def __repr__(self):
        return (f"Inventory with {self.num_machines} identical machines, MTTF={self.mttf}, a={self.degradation_a}"
                f" Ch={1}, Co={self._ordering_cost}, and Ce={self._emergency_cost}.")

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        random.seed(seed)

        # Reset timing, required after learning
        self.time_step = 1

        # Reset degradations, inventory and outstanding outstanding_orders
        self.degradations = np.zeros((self.num_machines,))
        self.inventory_level = 0
        self.outstanding_orders = 0

        # Reset costs
        self._holding_total = 0.
        self._ordering_total = 0.
        self._emergency_total = 0.
        self._total_cost = 0.

        self._average_stock = self.inventory_level
        self.total_maintenance = 1
        self.total_expedited_orders = 0

        if self.deterministic_lead_time:
            self.orders_pipeline.reset()
        if self.batch_ordering:
            self.orders_pipeline.reset()

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        assert self.outstanding_orders >= 0, f"Outstanding orders are negative"
        assert self.inventory_level >= 0, f"Outstanding orders are negative"
        assert action >= 0, f"Action {action} orders are negative"
        action = int(action)
        step_costs = 0.

        # Ensure action is doable and does not violate capacity constraints
        assert action + self.inventory_level + self.outstanding_orders <= self.inventory_capacity, \
            (f"Decision {action} leads to exceeding capacity:\n"
             f"D + I + O = {action} + {self.inventory_level} + {self.outstanding_orders} > {self.inventory_capacity}")

        # Spare Parts arrive: Update On and In
        if self.batch_ordering or self.deterministic_lead_time:
            arrivals = self.orders_pipeline.get_arrivals()
        else:
            arrivals = self.np_random.binomial(n=self.outstanding_orders, p=self.bino_prob)
        self.outstanding_orders -= int(arrivals)
        self.inventory_level += int(arrivals)

        # Update Outstanding orders with new orders
        step_costs += self._update_orders(action)

        # Update Degradation
        self.degradations += self.np_random.gamma(self.degradation_a, 1 / self.degradation_u, self.num_machines)
        # Perform Maintenance
        step_costs += self._perform_maintenance()

        holding_cost = self.inventory_level * self._holding_cost
        step_costs += holding_cost
        self._holding_total += holding_cost

        # Update E[S]
        if self.time_step > 0:
            self._average_stock = (self._average_stock * (self.time_step - 1) +
                                   self.inventory_level) / (self.time_step)
        self.time_step += 1

        self._total_cost = self._emergency_total + self._ordering_total + self._holding_total

        assert type(self.inventory_level) in [int, np.int64], (
            f"Inventory level is of type {type(self.inventory_level)} instead of"
            f" int | Step={self.time_step}")
        assert type(self.outstanding_orders) in [int, np.int64]

        obs = self._get_obs()
        info = self._get_info()

        return obs, -step_costs / self.max_cost, False, False, info

    def _update_orders(self, action: int) -> float:
        if action == 0:
            return 0.
        self.outstanding_orders += action
        self.orders_pipeline.add_order(action) if (self.batch_ordering or self.deterministic_lead_time) else None
        cost = self._ordering_cost if self.batch_ordering else (self._ordering_cost * action)
        self._ordering_total += cost
        return cost

    def _perform_maintenance(self) -> float:
        machine_idx = np.where(self.degradations >= self._maintenance_threshold)[0]
        repairs = len(machine_idx)
        self.total_maintenance += repairs
        if repairs == 0:
            return 0.
        else:
            self.degradations[machine_idx] = 0
            expedites_needed = max(repairs - self.inventory_level, 0)
            self.inventory_level = max(0, self.inventory_level - repairs)
            arrivals = 0
            if expedites_needed > 0:
                self.total_expedited_orders += expedites_needed
                if self.deterministic_lead_time or self.batch_ordering:
                    arrivals = self.orders_pipeline.expedite_arrivals(expedites_needed)
                    if arrivals > 0:
                        self.inventory_level += max(arrivals - expedites_needed, 0)
                self.outstanding_orders = max(0, self.outstanding_orders - expedites_needed)
                self._emergency_total += expedites_needed * self._emergency_cost
                self._ordering_total +=  (expedites_needed - arrivals) * self._ordering_cost
            return expedites_needed * self._emergency_cost + (expedites_needed - arrivals) * self._ordering_cost

    def action_masks(self):
        """
        Return an action mask with the allowable action having a True
        :return:
        """
        # mask = self._action_array <= max(0, self.inventory_capacity
        #                                  - self.inventory_level
        #                                  - self.outstanding_orders)
        mask = self._action_array <= min(MAX_BATCH_SIZE, max(0, self.inventory_capacity - self.inventory_level -
                                                             np.sum(self.parts_array * self.outstanding_orders)))
        return mask.astype(dtype=bool)

    def _get_obs(self):
        if self.sorted_degradation:
            array = np.append(np.sort(self.degradations) / self._maintenance_threshold,
                             [self.inventory_level / self.inventory_capacity,
                              *self.orders_pipeline.get_state() / self.inventory_capacity])
        else:
            array = np.append(self.degradations / self._maintenance_threshold,
                              [self.inventory_level / self.inventory_capacity,
                               *self.orders_pipeline.get_state() / self.inventory_capacity])
        return array.astype(np.float32)

    def _get_info(self):
        fill_rate = 1 - self.total_expedited_orders / self.total_maintenance
        return {
            "time_step": self.time_step,
            "total_cost": self._total_cost,
            "average_cost": self._total_cost / self.time_step,
            "holding_costs": self._holding_total,
            "ordering_costs": self._ordering_total,
            "emergency_costs": self._emergency_total,
            "average_inventory": self._average_stock,
            "fill_rate": fill_rate
        }


class InventoryRewardShaping(Inventory):
    """
    This subclass enables integrating reward shaping by using either the BSP or ProBSP as a teacher policy.
    By penalizing decisions that are far from these policies, the agent will learn to focus on more relevant decision,
    while still exploring potentially better decision.
    """
    def __init__(self,
                 machines: int = 1,
                 lead_time_p: float = 1,
                 mttf: float = 10.,
                 a: float = 1.,
                 ordering_cost: float = 2,
                 emergency_cost: float = 5,
                 deterministic_lead_time: bool = False,
                 sorted_degradation: bool = True,
                 bsp: bool = True,
                 probsp: bool = False,
                 bsp_n: int = None,
                 probsp_n: int = None,
                 probsp_xo: float = None,
                 gamma:float = None):
        super().__init__(machines=machines,
                         lead_time_p=lead_time_p,
                         mttf=mttf,
                         a=a,
                         ordering_cost=ordering_cost,
                         emergency_cost=emergency_cost,
                         deterministic_lead_time=deterministic_lead_time,
                         sorted_degradation=sorted_degradation)
        if bsp and probsp:
            raise AssertionError("Choose only to use BSP or ProBSP for Reward Shaping")
        if bsp and (bsp_n is None):
            raise ValueError("Should specify the value of BSP initial stock level")
        if probsp:
            if probsp_n is None or probsp_xo is None:
                raise ValueError(f"Expected to have N, and Xo values, but got {probsp_n}, {probsp_xo}")
        if gamma is None or gamma > 1 or gamma < 0:
            raise ValueError(f"Expected a value of discount factor (gamma) between [0,1], got {gamma} instead")
        self.use_bsp = bsp
        self.use_probsp = probsp
        self.bsp_n = bsp_n
        self.probsp_n = probsp_n
        self.probsp_xo = probsp_xo
        self.penalty = 0.0001
        self.gamma = gamma
        self._previous_potential = 0

    def copy(self):
        dummy = InventoryRewardShaping(
            self.num_machines, self.lead_time_p, self.mttf, self.degradation_a,
            ordering_cost=self._ordering_cost, emergency_cost=self._emergency_cost,
            deterministic_lead_time=self.deterministic_lead_time,
            bsp=self.use_bsp, probsp=self.use_probsp,
            bsp_n=self.bsp_n, probsp_n=self.probsp_n, probsp_xo=self.probsp_xo, gamma=self.gamma
        )

    def get_rs_decision(self):
        if self.use_bsp:
            decision = self.bsp_n - self.inventory_level - self.outstanding_orders
        elif self.use_probsp:
            decision = (self.probsp_n + np.sum(self.degradations > self.probsp_xo)
                        - self.inventory_level - self.outstanding_orders)
        else:
            raise ValueError("Something wrong")
        return decision

    def _compute_penalty(self):
        cur_potential = 0.0001 * abs(self.get_rs_decision())
        penalty = self.gamma * cur_potential - self._previous_potential
        self._previous_potential = cur_potential
        return -penalty

    def step(self, action: int, verbose: bool = False):
        obs, costs, terminated, truncated, info = super().step(action)
        costs += self._compute_penalty()
        return obs, costs, terminated, truncated, info


class TestInventory(Inventory):
    """
    This subclass enables integrating reward shaping by using either the BSP or ProBSP as a teacher policy.
    By penalizing decisions that are far from these policies, the agent will learn to focus on more relevant decision,
    while still exploring potentially better decision.
    """
    def __init__(self,
                 machines: int = 1,
                 lead_time_p: float = 1,
                 mttf: float = 10.,
                 a: float = 1.,
                 ordering_cost: float = 2,
                 emergency_cost: float = 5,
                 deterministic_lead_time: bool = False,
                 sorted_degradation: bool = True,
                 ):
        super().__init__(machines=machines,
                         lead_time_p=lead_time_p,
                         mttf=mttf,
                         a=a,
                         ordering_cost=ordering_cost,
                         emergency_cost=emergency_cost,
                         deterministic_lead_time=deterministic_lead_time,
                         sorted_degradation=sorted_degradation)
        self.outstanding_orders = np.zeros(MAX_BATCH_SIZE, dtype=int)

    def copy(self):
        dummy = TestInventory(self.num_machines, self.lead_time_p, self.mttf, self.degradation_a,
                              ordering_cost=self._ordering_cost, emergency_cost=self._emergency_cost,
                              deterministic_lead_time=self.deterministic_lead_time)
        return dummy

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        # We need the following line to seed self.np_random
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
        # Reset timing, required after learning
        self.time_step = 1

        # Reset degradations, inventory and outstanding outstanding_orders
        self.degradations = np.zeros((self.num_machines,))
        self.inventory_level = 0
        self.outstanding_orders = np.zeros(MAX_BATCH_SIZE)

        # Reset costs
        self._holding_total = 0.
        self._ordering_total = 0.
        self._emergency_total = 0.
        self._total_cost = 0.

        self._average_stock = self.inventory_level
        self.total_maintenance = 1
        self.total_expedited_orders = 0

        if self.deterministic_lead_time:
            self.orders_pipeline.reset()
        if self.batch_ordering:
            self.orders_pipeline.reset()

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        assert self.inventory_level >= 0, f"Outstanding orders are negative"
        assert action >= 0, f"Action {action} orders are negative"
        action = int(action)
        step_costs = 0.

        # Ensure action is doable and does not violate capacity constraints
        assert action + self.inventory_level + np.sum(self.outstanding_orders * self.parts_array) <= self.inventory_capacity, \
            (f"Decision {action} leads to exceeding capacity:\n"
             f"D + I + O = {action} + {self.inventory_level} + {self.outstanding_orders} > {self.inventory_capacity}")

        # Get arriving parts:
        orders_arrivals = self.np_random.binomial(n=np.sum(self.outstanding_orders), p=self.bino_prob)
        appended_batches = np.repeat(self.parts_array, self.outstanding_orders.tolist())
        sampled_batches = self.np_random.choice(appended_batches, size=orders_arrivals, replace=False)
        for batch in sampled_batches:
            self.outstanding_orders[batch-1] -= 1
            if self.outstanding_orders[batch - 1] < 0:
                raise ValueError
            self.inventory_level += batch
        if action > 0:
            self.outstanding_orders[action-1] += 1
            step_costs += self._ordering_cost
            self._ordering_total += self._ordering_cost
        # Update Degradation
        self.degradations += self.np_random.gamma(self.degradation_a, 1 / self.degradation_u, self.num_machines)
        # Perform Maintenance
        step_costs += self._perform_maintenance()

        holding_cost = self.inventory_level * self._holding_cost
        step_costs += holding_cost
        self._holding_total += holding_cost

        # Update E[S]
        if self.time_step > 0:
            self._average_stock = (self._average_stock * (self.time_step - 1) +
                                   self.inventory_level) / (self.time_step)
        self.time_step += 1

        self._total_cost = self._emergency_total + self._ordering_total + self._holding_total

        assert type(self.inventory_level) in [int, np.int64], (
            f"Inventory level {self.inventory_level} is of type {type(self.inventory_level)} instead of"
            f" int | Step={self.time_step}")

        obs = self._get_obs()
        info = self._get_info()

        return obs, -step_costs / self.max_cost, False, False, info

    def _perform_maintenance(self) -> float:
        machine_idx = np.where(self.degradations >= self._maintenance_threshold)[0]
        repairs = len(machine_idx)
        self.total_maintenance += repairs
        if repairs == 0:
            return 0.
        else:
            self.degradations[machine_idx] = 0
            expedites_needed = max(repairs - self.inventory_level, 0)
            self.inventory_level = max(0, self.inventory_level - repairs)
            arrivals = 0
            if expedites_needed > 0:
                self.total_expedited_orders += expedites_needed
                arrivals = self.expedite_arrivals(expedites_needed, partial_expedite=True)
                self.inventory_level += max(arrivals - expedites_needed, 0)
                self._emergency_total += expedites_needed * self._emergency_cost
                self._ordering_total += (expedites_needed - arrivals) * self._ordering_cost
            return expedites_needed * self._emergency_cost + (expedites_needed - arrivals) * self._ordering_cost

    def expedite_arrivals(self, expedites_needed, partial_expedite=True) -> int:
        if partial_expedite:
            arrivals = 0
            appended_batches = np.repeat(self.parts_array, self.outstanding_orders.tolist())
            appended_batches = appended_batches.tolist()
            if arrivals < expedites_needed and np.sum(appended_batches) > 0:
                _size = min(expedites_needed, len(appended_batches))
                sampled_batch = self.np_random.choice(appended_batches, size=_size, replace=False)
                arrivals = _size
                # Because of partial expedition, if an order of size x is partially expedited, batch x-1 is increased
                #  unless x = 1
                for batch in sampled_batch:
                    self.outstanding_orders[batch - 1] -= 1
                    if batch > 1:
                        self.outstanding_orders[batch - 2] += 1
            return arrivals
        else:
            appended_batches = np.repeat(self.parts_array, self.outstanding_orders.tolist())
            appended_batches = appended_batches.tolist()
            arrivals = 0
            while arrivals < expedites_needed and np.sum(appended_batches) > 0:
                sampled_batch = self.np_random.choice(appended_batches, size=1, replace=False)
                arrivals += int(sampled_batch)
                self.outstanding_orders[sampled_batch - 1] -= 1
                if self.outstanding_orders[sampled_batch - 1] < 0:
                    raise ValueError
                appended_batches.remove(sampled_batch)
        return arrivals

    def action_masks(self):
        mask = self._action_array <= min(MAX_BATCH_SIZE, max(0, self.inventory_capacity - self.inventory_level -
                                                             np.sum(self.parts_array * self.outstanding_orders)))
        return mask.astype(dtype=bool)

    def _get_obs(self):
        if self.sorted_degradation:
            array = np.append(np.sort(self.degradations) / self._maintenance_threshold,
                             [self.inventory_level / self.inventory_capacity,
                              *self.outstanding_orders /  self.inventory_capacity])
        else:
            array = np.append(self.degradations / self._maintenance_threshold,
                              [self.inventory_level / self.inventory_capacity,
                               *self.outstanding_orders / self.inventory_capacity])
        return array.astype(np.float32)

