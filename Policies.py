


from typing import Union

import numpy as np
from Environments import Inventory


class ProBSP:
    """
        A Threshold Policy for inventory management.

        Attributes:
            xo (float): The degradation threshold (as a fraction) for ordering spare parts.
            n (int): The base stock level.
            capacity (int): The maximum inventory capacity of the environment.
            machines (int): The number of machines in the environment.
        """
    def __init__(self, env: Union[Inventory], n,xo):
        """
            Initializes the ProBSP policy.

            Args:
                inventory (Inventory): The inventory environment instance.
                n (int): The base stock level adjustment factor.
                xo (float): The degradation threshold (percentage) for ordering spare parts.
        """
        self.xo = xo / 100
        self.n = n
        self.capacity = env.inventory_capacity
        self.machines = env.num_machines

    def __repr__(self):
        return f"Threshold Policy with N={self.n} and Xo={self.xo * 100}"

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
        outstanding = obs[-1] * self.capacity
        inventory = obs[-2] * self.capacity
        demand = np.sum(degradations >= self.xo, dtype=int)
        action = demand + self.n - outstanding - inventory
        # action = np.round(action)
        action = min(action, self.capacity - outstanding - inventory)
        return action.astype(int), None


class BaseStockPolicy:
    """
        A Base Stock Policy for inventory management.

        Attributes:
            stock (int): The base stock level.
            max_capacity (int): The maximum inventory capacity of the environment.
    """
    def __init__(self, env: Inventory, bs_level):
        """
                Initializes the Base Stock Policy.

                Args:
                    env (Inventory): The inventory environment instance.
                    bs_level (int): The base stock level.

                Raises:
                    AssertionError: If the base stock level exceeds the maximum capacity.
            """
        self.stock = bs_level
        self.max_capacity = env.inventory_capacity
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
        cur_stock = obs[-2] * self.max_capacity
        cur_outstanding = obs[-1] * self.max_capacity
        inventory_position = cur_stock + cur_outstanding
        action = self.stock - inventory_position
        # if np.round(action + inventory_position) > self.stock:
        #     raise ValueError(f"Got Action that would lead to excess stock"
        #                      f" action={action}, IP={inventory_position}")
        # if action < 0:
        #     raise ValueError(f"Action={action} cannot be negative, obs={obs}"
        #                      f" Policy={self}")
        return action, None
