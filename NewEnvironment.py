import random
from typing import Tuple, Union
import gym as gym
import numpy as np
from gym import spaces
from gym.core import ActType, ObsType


class GeometricOrderPipeline:
    """
       Order pipeline with geometric arrival process.

       Models spare parts orders in transit where each order has a fixed probability
       of arriving in each time step, resulting in a geometric distribution for lead times.

       Attributes:
           p (float): Probability that an order arrives in any given time step.
           capacity (int): Maximum number of concurrent orders that can be in the pipeline.
           pipeline (np.ndarray): Array tracking order quantities at each position.
           outstanding_parts (int): Total number of parts currently in the pipeline.
           seed (int, optional): Random seed for reproducibility.
   """
    def __init__(self, capacity: int, p: float, seed=None):
        """
            Initialize the geometric order pipeline.

            Args:
                capacity: Maximum number of orders that can be outstanding simultaneously.
                p: Arrival probability per time step (0 < p <= 1).
                seed: Random seed for reproducibility. Defaults to None.
        """
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
        """
        Add a new order to the pipeline.

        Args:
            a: Quantity of parts to order.

        Raises:
            AssertionError: If adding the order would exceed pipeline capacity.
        """
        assert self.outstanding_parts + a <= self.capacity, (f"Exceeded capacity: "
                                                             f"{self.outstanding_parts} + {a} > {self.capacity}")
        if self.outstanding_parts == self.capacity:
            return
        idx = np.where(self.pipeline == 0)[0][0]
        self.pipeline[idx] += a
        self.outstanding_parts += a

    def get_arrivals(self):
        """
        Process arrivals for the current time step using Bernoulli trials.

        Each order in the pipeline has probability p of arriving. Orders that
        arrive are removed from the pipeline and their quantities are summed.

        Returns:
            int: Total quantity of parts that arrived in this time step.
        """
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

        self.outstanding_parts -= arrivals
        return arrivals

    def expedite_arrivals(self, expedites_needed: int):
        """
        Expedite orders from the pipeline for emergency situations.

        Pulls parts from the pipeline immediately, starting with the oldest orders.
        Used when urgent demand exceeds available inventory.

        Args:
            expedites_needed: Number of parts needed immediately.

        Returns:
            int: Actual number of parts expedited (may be less if insufficient outstanding).

        Raises:
            AssertionError: If expedites_needed is not positive.
        """
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


class BaseClassOrderPipeline:
    """
    Base class for order pipelines.

    Models spare parts orders where each order has a specific remaining time
    until arrival. **Subclasses implement different lead time distributions**.

    Attributes:
        capacity (int): Maximum number of concurrent orders in the pipeline.
        pipeline (np.ndarray): Array of order quantities at each position.
        remaining_time (np.ndarray): Time steps remaining until each order arrives.
        outstanding_parts (int): Total parts currently in the pipeline.
    """
    def __init__(self, capacity: int):
        """
        Initialize the base order pipeline.

        Args:
            capacity: Maximum number of orders that can be outstanding simultaneously.
        """
        self.capacity = capacity
        self.pipeline = np.zeros(capacity, dtype=int)
        self.remaining_time = np.ones(capacity, dtype=int) * -1
        self.outstanding_parts = 0

    def __len__(self):
        return self.outstanding_parts

    def add_order(self, a:int):
        """
        Add a new order to the pipeline with a sampled lead time.

        Args:
            a: Quantity of parts to order.

        Raises:
            AssertionError: If adding the order would exceed capacity.
            ValueError: If capacity constraint is violated.
        """
        assert self.outstanding_parts + a <= self.capacity, (f"Exceeded capacity: On={self.pipeline}"
                                                             f"{self.outstanding_parts} + {a} > {self.capacity}")
        if self.outstanding_parts + a > self.capacity:
            raise ValueError(f"Cannot add order to pipeline due to capacity."
                             f"On={self.pipeline} and d_n={a}")
        if a > 0:
            first_idx = np.where(self.remaining_time == -1)[0][0]
            self.pipeline[first_idx] = a
            # TODO: check
            self.remaining_time[first_idx] = self._get_lead_time()-1
            self.outstanding_parts += a
        self._check()

    def get_arrivals(self):
        """
        Process arrivals for the current time step.

        Decrements remaining time for all orders. Orders with remaining_time == 0
        arrive and are removed from the pipeline.

        Returns:
            int: Total quantity of parts that arrived in this time step.
        """
        arrivals = 0
        for i in range(self.capacity):
            if self.remaining_time[i] == 0 and self.pipeline[i] > 0:
                arrivals += self.pipeline[i]
                # self.pipeline[i] = 0
                self.remaining_time[i] -= 1
            elif self.pipeline[i] > 0:
                self.remaining_time[i] -= 1
        self._compactify()
        self.outstanding_parts -= arrivals
        self._check()
        return arrivals

    def expedite_arrivals(self, expedites_needed:int):
        """
        Expedite orders from the pipeline for emergency situations.

        Pulls parts from the pipeline immediately, regardless of remaining time.

        Args:
            expedites_needed: Number of parts needed immediately.

        Returns:
            int: Actual number of parts expedited.

        Raises:
            AssertionError: If expedites_needed is not positive.
        """
        assert expedites_needed > 0, f"Number of expedites should be positive"
        arrivals = 0
        r = expedites_needed
        if self.outstanding_parts > 0:
            for i in range(self.capacity):
                if self.pipeline[i] > 0 and self.remaining_time[i] > -1:
                    if self.pipeline[i] <= r:
                        r -= self.pipeline[i]
                        self.pipeline[i] = 0
                        self.remaining_time[i] = -1
                    else:
                        self.pipeline[i] -= r
                        r = 0
                        break
            arrivals = expedites_needed - r
            self.outstanding_parts -= arrivals
            self._check()
            self._compactify()
            self._check()
            return arrivals
        else:
            return arrivals

    def _check(self):
        if self.outstanding_parts != np.sum(self.pipeline):
            raise ValueError("There was a discrepancy in the outstanding orders handling")

    def copy(self):
        pass

    def reset(self):
        self.pipeline = np.zeros(self.capacity, dtype=int)
        self.remaining_time = np.ones(self.capacity, dtype=int) * -1
        self.outstanding_parts = 0

    def _get_lead_time(self):
        """
        Sample a lead time for a new order. To be implemented by subclasses.

        Returns:
            int: Lead time in time steps.
        """
        pass

    def _compactify(self):
        """Compact the pipeline by removing empty slots and moving active orders forward."""

        _temp_pipeline = np.zeros_like(self.pipeline)
        _temp_remaining_time = np.ones_like(self.remaining_time) * -1
        _idx = 0
        for i in range(self.capacity):
            if self.pipeline[i] > 0 and self.remaining_time[i] >= 0:
                _temp_pipeline[_idx] = self.pipeline[i]
                _temp_remaining_time[_idx] = self.remaining_time[i]
                _idx += 1
        self.pipeline = _temp_pipeline
        self.remaining_time = _temp_remaining_time


class DeterministicOrderPipeline(BaseClassOrderPipeline):
    """
    Order pipeline with deterministic (fixed) lead times.

    All orders take exactly the same amount of time to arrive.

    Attributes:
        lead_time (int): Fixed lead time for all orders in time steps.
    """
    def __init__(self, capacity:int, lead_time:int):
        super().__init__(capacity=capacity)
        self.lead_time = lead_time

    def _get_lead_time(self):
        return self.lead_time


class UniformOrderPipeline(BaseClassOrderPipeline):
    """
    Order pipeline with uniformly distributed lead times.

    Lead times are sampled uniformly from a discrete range.

    Attributes:
        lead_times (list): List of possible lead time values.
    """
    def __init__(self, capacity:int, lower_lead_time:int, upper_lead_time:int):
        super().__init__(capacity)
        self.lead_times = [x for x in range(lower_lead_time, upper_lead_time+1)]

    def _get_lead_time(self):
        return random.choice(self.lead_times)

class EmpiricalOrderPipeline(BaseClassOrderPipeline):
    """
    Order pipeline with empirically-defined lead time distribution.

    Lead times are sampled according to user-specified probabilities,
    allowing for arbitrary discrete distributions based on historical data.

    Attributes:
        lead_times (Union[list, np.array]): Possible lead time values.
        lead_time_dist (Union[list, np.array]): Probability weights for each lead time.
    """
    def __init__(self, capacity: int, lead_times:Union[list, np.array], lead_time_dist:Union[list, np.array]):
        super().__init__(capacity)
        self.lead_time_dist = lead_time_dist
        self.lead_times = lead_times

    def _get_lead_time(self):
        return random.choices(population=self.lead_times, weights=self.lead_time_dist, k=1)[0]



class Inventory(gym.Env):
    """
   Spare parts inventory management environment with degrading machines.

   A gym environment simulating spare parts inventory management for machines
   that degrade over time. Machines require maintenance when degradation reaches
   a threshold, consuming spare parts from inventory. The agent must balance
   holding costs, ordering costs, and emergency expediting costs.

   State space includes:
       - Machine degradation levels (normalized)
       - Current inventory level (normalized)
       - Outstanding orders in pipeline (normalized)

   Action space:
       - Discrete: order quantity from 0 to max(inventory_capacity,batch_size)

   Reward:
       - Negative cost per time step (holding + ordering + emergency)
       - Normalized by maximum possible cost

   Attributes:
       num_machines (int): Number of machines being maintained.
       mttf (float): Mean time to failure parameter for degradation.
       degradation_a (float): Shape parameter for gamma degradation process.
       order_pipeline: Pipeline managing orders in transit.
       max_batch_size (int): Maximum parts that can be ordered at once.
       ordering_cost (float): ordering cost for a batch
       emergency_cost (float): emergency cost for expediting a **part**
   """
    def __init__(self,
                 machines: int,
                 order_pipeline: Union[GeometricOrderPipeline, BaseClassOrderPipeline,
                 DeterministicOrderPipeline, UniformOrderPipeline, EmpiricalOrderPipeline],
                 max_batch_size: int = 3,
                 mttf: float = 10.,
                 a: float = 1.,
                 ordering_cost: float = 2,
                 emergency_cost: float = 5,
                 sorted_degradation: bool = False,
                 ):
        """
        Initialize the inventory environment.

        Args:
            machines: Number of machines to maintain.
            order_pipeline: Pipeline object managing order lead times.
            max_batch_size: Maximum quantity per order. Defaults to 3.
            mttf: Mean time to failure for degradation process. Defaults to 10.
            a: Shape parameter for gamma degradation. Defaults to 1.
            ordering_cost: Cost per regular order placed. Defaults to 2.
            emergency_cost: Cost per part expedited. Defaults to 5.
            sorted_degradation: If True, sort degradation levels in observation. Defaults to False.
        """
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
        self._average_order_size = 0.
        self._number_of_orders = 0

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
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility. Defaults to None.
            options: Additional options (unused). Defaults to None.

        Returns:
            tuple: (observation, info) where observation is the initial state
                   and info contains metrics like costs and fill rate.
        """
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

        self._average_order_size = 0.
        self._number_of_orders = 0

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action: ActType, *args, **kwargs) -> Tuple[ObsType, float, bool, bool, dict]:
        """
        Execute one time step in the environment, given an *action* from the decision maker

        Processes: order arrivals, new order placement, degradation updates,
        maintenance actions, and cost calculations.

        Args:
            action: Order quantity to place (0 to inventory_capacity).

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
                - observation: Current state after step
                - reward: Negative normalized cost for this step
                - terminated: Always False
                - truncated: Always False
                - info: Dictionary with cost breakdown and metrics

        Raises:
            AssertionError: If action violates capacity constraints or is negative.
        """
        assert self.inventory_level >= 0, f"Outstanding orders are negative"
        assert action >= 0, (f"Action {action} orders are negative \n"
                             f"I={self.inventory_level}, On={self.order_pipeline.pipeline}")
        action = int(action)

        if action > 0:
            self._average_order_size = ((self._number_of_orders * self._average_order_size + action) /
                                        (self._number_of_orders + 1))
            self._number_of_orders += 1

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
        info['step_cost'] = step_costs
        return obs, -step_costs / self.max_cost, False, False, info

    def _get_obs(self, normalized=True):
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
            "holding_costs": self._holding_total/ self.time_step,
            "ordering_costs": self._ordering_total/ self.time_step,
            "emergency_costs": self._emergency_total/ self.time_step,
            "average_inventory": self._average_stock,
            "average_order_size": self._average_order_size,
            "fill_rate": fill_rate
        }

    def _perform_maintenance(self):
        """
        Perform maintenance on machines exceeding degradation threshold.

        Consumes spare parts from inventory. If insufficient inventory,
        expedites orders from pipeline and incurs emergency costs.

        Returns:
            float: Total cost incurred from maintenance (emergency + ordering).
        """
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
        Return an action mask with the allowable action having a True value
        :return:
        """

        mask = self._action_array <= min(self.max_batch_size, max(0, self.inventory_capacity - self.inventory_level -
                                                             self.order_pipeline.outstanding_parts))
        return mask.astype(dtype=bool)


class InventoryRS(Inventory):
    """
    Inventory environment with reward shaping.

    Extends the base Inventory environment with potential-based reward shaping
    based on either a Base-Stock Policy (BSP) or Probabilistic Base-Stock Policy (ProBSP).
    Adds a penalty term to the reward that encourages following the heuristic policy.

    Attributes:
        use_bsp (bool): Whether to use Base-Stock Policy for shaping.
        use_probsp (bool): Whether to use Proactive Base-Stock Policy for shaping.
        bsp_n (int): Target stock level for BSP.
        probsp_n (int): Base target level for ProBSP.
        probsp_xo (float): Degradation threshold for ProBSP adjustment.
        gamma (float): Discount factor for potential-based shaping.
    """
    def __init__(self,
                 machines: int,
                 order_pipeline: Union[GeometricOrderPipeline, BaseClassOrderPipeline,
                 DeterministicOrderPipeline, UniformOrderPipeline, EmpiricalOrderPipeline],
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
        Initialize reward-shaped inventory environment.

        Args:
            machines: Number of machines to maintain.
            order_pipeline: Pipeline object managing order lead times.
            max_batch_size: Maximum quantity per order. Defaults to 3.
            mttf: Mean time to failure. Defaults to 10.
            a: Degradation shape parameter. Defaults to 1.
            ordering_cost: Cost per regular order. Defaults to 2.
            emergency_cost: Cost per expedited part. Defaults to 5.
            sorted_degradation: Sort degradation in observation. Defaults to False.
            bsp: Use Base-Stock Policy for shaping. Defaults to True.
            probsp: Use Proactive Base-Stock Policy. Defaults to False.
            bsp_n: Target stock level for BSP. Required if bsp=True.
            probsp_n: Base target level for ProBSP. Required if probsp=True.
            probsp_xo: Degradation threshold for ProBSP. Required if probsp=True.
            gamma: Discount factor (0-1). Required.

        Raises:
            AssertionError: If both bsp and probsp are True.
            ValueError: If required parameters are missing or invalid.
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
        """
        Compute the recommended action according to the chosen heuristic.

        For BSP: Orders up to target level (bsp_n - inventory - outstanding).
        For ProBSP: Adjusts target based on number of machines above degradation threshold.

        Returns:
            int: Recommended order quantity constrained by capacity and batch size.

        Raises:
            ValueError: If neither BSP nor ProBSP is enabled.
        """
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
        """
        Compute the potential-based reward shaping penalty.

        Penalizes deviation from the heuristic policy decision. Uses potential-based
        shaping to maintain optimal policy invariance.

        Args:
            action: The action taken by the agent.

        Returns:
            float: Shaped reward adjustment (negative of penalty).
        """
        cur_potential = 0.0001 * abs(self.get_rs_decision() - action)
        penalty = self.gamma * cur_potential - self._previous_potential
        self._previous_potential = cur_potential
        return -penalty

    def step(self, action: int, verbose: bool = False):
        """
        Execute one time step with reward shaping.

        Extends parent step() by adding potential-based shaping penalty.

        Args:
            action: Order quantity to place.
            verbose: If True, print debug information. Defaults to False.

        Returns:
            tuple: (observation, shaped_reward, terminated, truncated, info)
        """
        obs, costs, terminated, truncated, info = super().step(action)
        costs += self._compute_penalty(action)
        return obs, costs, terminated, truncated, info


class InventoryAggregated_NOTUSED(Inventory):
    """
    Inventory environment with aggregated order pipeline observation.

    Differs from base Inventory by representing the order pipeline as aggregated
    counts of orders by batch size rather than individual order positions. This
    reduces observation space dimensionality when max_batch_size < inventory_capacity.

    The pipeline observation contains max_batch_size elements, where element i
    represents the count of outstanding orders of size (i+1).

    Attributes:
        All attributes from parent Inventory class.
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
                 ):
        """
        Initialize aggregated inventory environment.

        Args:
            machines: Number of machines to maintain.
            order_pipeline: Pipeline object managing order lead times.
            max_batch_size: Maximum quantity per order. Defaults to 3.
            mttf: Mean time to failure. Defaults to 10.
            a: Degradation shape parameter. Defaults to 1.
            ordering_cost: Cost per regular order. Defaults to 2.
            emergency_cost: Cost per expedited part. Defaults to 5.
            sorted_degradation: Sort degradation in observation. Defaults to False.
        """
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
        dummy = InventoryAggregated_NOTUSED(
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
