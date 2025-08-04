import random
from typing import Tuple

import gym as gym
import numpy as np
from gym import spaces
from gym.core import ActType, ObsType


class GeometricOrderPipeline:

    def __init__(self, capacity: int, p: float, seed=None):
        self.p = p
        self.capacity = capacity
        self.pipeline = np.zeros(capacity, dtype=int)
        self.outstanding_parts = 0

        if seed is not None:
            np.random.seed(seed)
            self.seed = seed
        self.samples = 100000
        self.bernoulli_rvs = np.random.binomial(1, p, (self.samples, capacity))
        self.rvs_idx = 0

    def __len__(self):
        return self.outstanding_parts

    def add_order(self, a: int):
        assert self.outstanding_parts + a <= self.capacity, f"Exceeded capacity"
        if self.outstanding_parts == self.capacity:
            return
        idx = np.where(self.pipeline == 0)[0][0]
        self.pipeline[idx] += a
        self.outstanding_parts += a

    def get_arrivals(self):
        arrivals_bool = self.bernoulli_rvs[self.rvs_idx]
        self.rvs_idx += 1
        self._check_rvs()
        arrivals = np.sum(arrivals_bool * self.pipeline)
        new_pipeline = np.zeros(self.capacity, dtype=int)
        j = 0
        for i in range(self.capacity):
            if self.pipeline[i] > 0 and not arrivals_bool[i]:
                new_pipeline[j] = self.pipeline[i]
                j += 1
        self.pipeline = new_pipeline

        # Vectorized compaction
        # remaining_mask = (self.pipeline > 0) & (~arrivals_bool)
        # remaining_orders = self.pipeline[remaining_mask.astype(bool)]
        #
        # self.pipeline.fill(0)
        # self.pipeline[:len(remaining_orders)] = remaining_orders

        self.outstanding_parts -= arrivals
        return arrivals

    def expedite_arrivals(self, expedites_needed: int):
        assert expedites_needed > 0, f"Number of expedites should be positive"
        if self.outstanding_parts > 0:
            r = expedites_needed
            for i in range(self.capacity):
                if self.pipeline[i] < r:
                    r -= self.pipeline[i]
                    self.pipeline[i] = 0
                else:
                    self.pipeline[i] -= r
                    r = 0
                    break
            arrivals = expedites_needed - r
            self.outstanding_parts -= arrivals
            return arrivals
        else:
            return 0

    def copy(self):
        dummy = GeometricOrderPipeline(
            capacity=self.capacity, p=self.p
        )
        return dummy

    def _check_rvs(self):
        if self.rvs_idx == self.samples - 1:
            self.rvs_idx = 0
            self.bernoulli_rvs = np.random.binomial(1, self.p, (self.samples, self.capacity))

    def reset(self):
        self.pipeline = np.zeros(self.capacity, dtype=int)
        self.outstanding_parts = 0

        self.samples = 100000
        self.bernoulli_rvs = np.random.binomial(1, self.p, (self.samples, self.capacity))
        self.rvs_idx = 0


class Inventory(gym.Env):

    def __init__(self,
                 machines: int,
                 order_pipeline: GeometricOrderPipeline,
                 max_batch_size: int = 3,
                 mttf: float = 10.,
                 a: float = 1.,
                 ordering_cost: float = 2,
                 emergency_cost: float = 5,
                 sorted_degradation: bool = False,
                 ):
        self.sorted_degradation = sorted_degradation
        self.inventory_capacity = machines
        self.max_batch_size = max_batch_size
        self._maintenance_threshold = 100
        self.num_machines = machines
        self.mttf = mttf
        self.degradation_a = a
        self.degradation_u = (mttf * a - 0.5) / 100
        self.batch_ordering = True
        self.order_pipeline = order_pipeline

        # cost components
        self._holding_cost = 1.
        self._ordering_cost = ordering_cost
        self._emergency_cost = emergency_cost

        self.max_cost = self._emergency_cost * self.num_machines

        # State
        self.degradations = np.zeros((self.num_machines,))
        self.inventory_level = 0
        self.outstanding_orders = np.zeros((self.num_machines,))

        # Total Costs
        # Computed Cumulative Costs
        self._holding_total = 0.
        self._ordering_total = 0.
        self._emergency_total = 0.
        self._total_cost = 0

        # Observation and Action space
        self.action_space = spaces.Discrete(self.inventory_capacity + 1)
        self._action_array = np.asarray([a for a in range(machines + 1)])
        deg_low = np.array([0.] * self.num_machines)
        deg_high = np.array([1.] * self.num_machines)
        inventory_low = np.array([0.])
        inventory_high = np.array([1.])
        outstanding_low = np.array([0.] * self.num_machines)
        outstanding_high = np.array([1.] * self.num_machines)
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

        self.total_expedited_orders = 0
        self.total_maintenance = 1

        self.time_step = 1
        self._average_stock = self.inventory_level

    def __str__(self):
        return "Inventory"

    def copy(self):
        dummy = Inventory(
            machines=self.num_machines,
            ordering_cost=self._ordering_cost,
            emergency_cost=self._emergency_cost,
            order_pipeline=self.order_pipeline.copy(),
            mttf=self.mttf,
            a=self.degradation_a,
            sorted_degradation=self.sorted_degradation
        )
        return dummy

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        random.seed(seed)

        self.order_pipeline.reset()

        # Reset timing, required after learning
        self.time_step = 1

        # Reset degradations, inventory and outstanding outstanding_orders
        self.degradations = np.zeros((self.num_machines,))
        self.inventory_level = 0
        self.outstanding_orders = self.order_pipeline.outstanding_parts

        # Reset costs
        self._holding_total = 0.
        self._ordering_total = 0.
        self._emergency_total = 0.
        self._total_cost = 0.

        self._average_stock = self.inventory_level
        self.total_maintenance = 1
        self.total_expedited_orders = 0

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        assert self.inventory_level >= 0, f"Outstanding orders are negative"
        assert action >= 0, (f"Action {action} orders are negative \n"
                             f"I={self.inventory_level}, On={self.order_pipeline.pipeline}")
        action = int(action)
        step_costs = 0.

        # Data stored for reward shaping implementation
        self._old_inventory_level = self.inventory_level
        self._old_degradation = self.degradations.copy()
        self._old_outstanding_orders = self.order_pipeline.outstanding_parts

        # Ensure action is doable and does not violate capacity constraints
        assert action + self.inventory_level + self.order_pipeline.outstanding_parts <= self.inventory_capacity, \
            (f"Decision {action} leads to exceeding capacity:\n"
             f"D + I + O = {action} + {self.inventory_level} + {self.order_pipeline.outstanding_parts} > "
             f"{self.inventory_capacity}\n"
             f"On={self.order_pipeline.pipeline}")

        # Spare Parts arrive: Update On and In
        arrivals = self.order_pipeline.get_arrivals()
        self.order_pipeline.add_order(action)
        self.inventory_level += int(arrivals)

        if action > 0:
            self._ordering_total += self._ordering_cost
            step_costs += self._ordering_cost

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

    def _get_obs(self):
        if self.sorted_degradation:
            array = np.append(np.sort(self.degradations) / self._maintenance_threshold,
                              [self.inventory_level / self.inventory_capacity,
                               *self.order_pipeline.pipeline / self.inventory_capacity])
        else:
            array = np.append(self.degradations / self._maintenance_threshold,
                              [self.inventory_level / self.inventory_capacity,
                               *self.order_pipeline.pipeline / self.inventory_capacity])
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

    def _perform_maintenance(self):
        machine_idx = np.where(self.degradations >= self._maintenance_threshold)[0]
        repairs = len(machine_idx)
        self.total_maintenance += repairs
        if repairs == 0:
            return 0.
        else:
            self.degradations[machine_idx] = 0
            expedites_needed = max(repairs - self.inventory_level, 0)
            self.total_expedited_orders += expedites_needed
            self.inventory_level = max(0, self.inventory_level - repairs)
            if expedites_needed > 0:
                expedited_arrivals = self.order_pipeline.expedite_arrivals(expedites_needed)
                self._emergency_total += expedites_needed * self._emergency_cost
                self._ordering_total += (expedites_needed - expedited_arrivals) * self._ordering_cost
                return expedites_needed * self._emergency_cost + (
                            expedites_needed - expedited_arrivals) * self._ordering_cost
            else:
                return 0

    def action_masks(self):
        """
        Return an action mask with the allowable action having a True
        :return:
        """
        # mask = self._action_array <= max(0, self.inventory_capacity
        #                                  - self.inventory_level
        #                                  - self.outstanding_orders)
        mask = self._action_array <= min(self.max_batch_size, max(0, self.inventory_capacity - self.inventory_level -
                                                             self.order_pipeline.outstanding_parts))
        return mask.astype(dtype=bool)


class InventoryRS(Inventory):
    """
    An inventory sub instance with reward shaping either from the BSP or ProBSP
    """
    def __init__(self,
                 machines: int,
                 order_pipeline: GeometricOrderPipeline,
                 max_batch_size:int = 3,
                 mttf: float = 10.,
                 a: float = 1.,
                 ordering_cost: float = 2,
                 emergency_cost: float = 5,
                 sorted_degradation: bool = False,
                 bsp: bool = True,
                 probsp: bool = False,
                 bsp_n: int = None,
                 probsp_n: int = None,
                 probsp_xo: float = None,
                 gamma: float = None
                 ):
        """

        :param machines:
        :param order_pipeline:
        :param max_batch_size:
        :param mttf:
        :param a:
        :param ordering_cost:
        :param emergency_cost:
        :param sorted_degradation:
        :param bsp:
        :param probsp:
        :param bsp_n:
        :param probsp_n:
        :param probsp_xo:
        :param gamma:
        """
        super().__init__(machines=machines, order_pipeline=order_pipeline, max_batch_size=max_batch_size,
                         mttf=mttf, a=a, ordering_cost=ordering_cost, emergency_cost=emergency_cost,
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
        self.bsp_n = int(bsp_n) if self.use_bsp else None
        self.probsp_n = int(probsp_n) if self.use_probsp else None
        self.probsp_xo = float(probsp_xo) if self.use_probsp else None
        self.penalty = 0.0001
        self.gamma = gamma
        self._previous_potential = 0

    def __str__(self):
        return "Inventory-RS"

    def copy(self):
        dummy = InventoryRS(
            machines=self.num_machines,
            order_pipeline=self.order_pipeline,
            max_batch_size=self.max_batch_size,
            mttf=self.mttf,
            a=self.degradation_a,
            ordering_cost=self._ordering_cost,
            emergency_cost=self._emergency_cost,
            sorted_degradation=self.sorted_degradation,
            bsp=self.use_bsp,
            probsp=self.use_probsp,
            bsp_n=self.bsp_n,
            probsp_n=self.probsp_n,
            probsp_xo=self.probsp_xo,
            gamma=self.gamma
        )
        return dummy

    def get_rs_decision(self):
        if self.use_bsp:
            decision = min(self.bsp_n - self._old_inventory_level - self._old_outstanding_orders, self.max_batch_size)
        elif self.use_probsp:
            decision = min((self.probsp_n + np.sum(self.degradations > self.probsp_xo)
                        - self._old_inventory_level - self._old_outstanding_orders),
                           self.inventory_capacity - self._old_outstanding_orders - self._old_inventory_level,
                           self.max_batch_size)
        else:
            raise ValueError("Something wrong")
        return decision

    def _compute_penalty(self, action):
        cur_potential = 0.0001 * abs(self.get_rs_decision() - action)
        penalty = self.gamma * cur_potential - self._previous_potential
        self._previous_potential = cur_potential
        return -penalty

    def step(self, action: int, verbose: bool = False):
        obs, costs, terminated, truncated, info = super().step(action)
        costs += self._compute_penalty(action)
        return obs, costs, terminated, truncated, info


class InventoryAggregated(Inventory):

    def __init__(self,
                 machines: int,
                 order_pipeline: GeometricOrderPipeline,
                 max_batch_size:int = 3,
                 mttf: float = 10.,
                 a: float = 1.,
                 ordering_cost: float = 2,
                 emergency_cost: float = 5,
                 sorted_degradation: bool = False,
                 ):
        super().__init__(machines=machines, order_pipeline=order_pipeline, max_batch_size=max_batch_size,
                         mttf=mttf, a=a, ordering_cost=ordering_cost, emergency_cost=emergency_cost,
                         sorted_degradation=sorted_degradation)
        deg_low = np.array([0.] * self.num_machines)
        deg_high = np.array([1.] * self.num_machines)
        inventory_low = np.array([0.])
        inventory_high = np.array([1.])
        outstanding_low = np.array([0.] * self.max_batch_size)
        outstanding_high = np.array([1.] * self.max_batch_size)
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

    def copy(self):
        dummy = InventoryAggregated(
            machines=self.num_machines,
            order_pipeline=self.order_pipeline,
            max_batch_size=self.max_batch_size,
            ordering_cost=self._ordering_cost,
            emergency_cost=self._emergency_cost,
            mttf=self.mttf,
            a=self.degradation_a,
            sorted_degradation=self.sorted_degradation
        )
        return dummy

    def __str__(self):
        return "Aggregated Inventory"

    def _get_obs(self):
        aggregated_order = [np.sum(self.order_pipeline.pipeline==i+1) for i in range(self.max_batch_size)]
        aggregated_order = np.array(aggregated_order, dtype=int)
        if self.sorted_degradation:
            array = np.append(np.sort(self.degradations) / self._maintenance_threshold,
                              [self.inventory_level / self.inventory_capacity,
                               *aggregated_order / self.inventory_capacity])
        else:
            array = np.append(self.degradations / self._maintenance_threshold,
                              [self.inventory_level / self.inventory_capacity,
                               *aggregated_order / self.inventory_capacity])
        return array.astype(np.float32)
