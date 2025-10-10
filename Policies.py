from typing import Union
import numpy as np
from NewEnvironment import Inventory

class ProBSP:
    """
        A Threshold Policy for inventory management.

        Attributes:
            xo (float): The degradation threshold (as a fraction) for ordering spare parts.
            n (int): The base stock level.
            capacity (int): The maximum inventory capacity of the environment.
            machines (int): The number of machines in the environment.
        """
    def __init__(self, env: Union[Inventory], n,xo, max_batch_size):
        """
            Initializes the ProBSP policy.

            Args:
                inventory (Inventory): The inventory environment instance.
                n (int): The base stock level adjustment factor.
                xo (float): The degradation threshold (percentage) for ordering spare parts.
                max_batch_size: the maximum size of a batch
        """
        self.xo = xo / 100
        self.n = n
        self.capacity = env.inventory_capacity
        self.machines = env.num_machines
        self.max_batch_size = max_batch_size

    def __repr__(self):
        return f"ProBSP with N={self.n} and Xo={self.xo * 100}"

    def predict(self, obs, *args, **kwargs):
        """
            Predicts the action to take based on the current observation.

            Args:
                obs (list or np.ndarray): The current state observation, including:
                    - Machine degradations (first `self.machines` elements, normalized).
                    - Inventory level (second-to-last element, normalized).
                    - Outstanding orders (last element, normalized).
                *args: Additional arguments (unused).
                **kwargs: Additional keyword arguments (unused).

            Returns:
                tuple: The action to take (int) and None (placeholder for compatibility).
            """
        degradations = obs[:self.machines]
        inventory = obs[self.machines] * self.capacity
        outstanding = np.sum(obs[self.machines+1:]) * self.capacity
        demand = np.sum(degradations >= self.xo)
        action = demand + self.n - outstanding - inventory
        action = min(action, self.capacity - outstanding - inventory, self.max_batch_size)
        action = np.round(action)
        return action, None


class BaseStockPolicy:
    """
        A Base Stock Policy for inventory management.

        Attributes:
            stock (int): The base stock level.
            max_capacity (int): The maximum inventory capacity of the environment.
    """
    def __init__(self, env: Inventory, bs_level, max_batch_size):
        """
                Initializes the Base Stock Policy.

                Args:
                    env (Inventory): The inventory environment instance.
                    bs_level (int): The base stock level.

                Raises:
                    AssertionError: If the base stock level exceeds the maximum capacity.
            """
        self.machines = env.num_machines
        self.stock = bs_level
        self.max_capacity = env.inventory_capacity
        self.max_batch_size = max_batch_size
        assert bs_level <= self.max_capacity, f"BS level cannot be higher than capacity"

    def __repr__(self):
        return f"Base Stock Policy with Level = {self.stock}"

    def predict(self, obs, *args, **kwargs):
        """
            Predicts the action to take based on the current observation.

            Args:
                obs (list or np.ndarray): The current state observation, including:
                    - Inventory level (second-to-last element, normalized).
                    - Outstanding orders (last element, normalized).
                *args: Additional arguments (unused).
                **kwargs: Additional keyword arguments (unused).

            Returns:
                tuple: The action to take (int) and None (placeholder for compatibility).

            Raises:
                ValueError: If the action would lead to excess stock or is negative.
        """
        inventory = obs[self.machines] * self.max_capacity
        outstanding = np.sum(obs[self.machines + 1:]) * self.max_capacity
        inventory_position = inventory + outstanding
        action = self.stock - inventory_position
        action = min(action, self.max_batch_size)
        return np.round(action), None
