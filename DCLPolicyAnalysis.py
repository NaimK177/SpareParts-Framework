"""
This file provides a tool to analyze the structure of the DCL policies learned.
It compares the imported DCL vs the ProBSP, to highlight the key differences in the structure
of the policy.

DCL Learned policies are expected in the directory 'dcl_data', and the corresponding
parameters can be set.
Not that the DCL does not required cost normalization, hence a subclass of Inventory is
written to facilitate the process
"""


import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from torch import nn
import torch.nn.functional as F
from MaskedDQN import MaskedDQN
from NewEnvironment import Inventory, GeometricOrderPipeline, UniformOrderPipeline, DeterministicOrderPipeline, EmpiricalOrderPipeline
from Auxiliaries import evaluate_policy, find_probsp
from Policies import ProBSP



class DQN2(nn.Module):

    def __init__(self, *args, **kwargs):
        super(DQN2, self).__init__()
        self.layer1 = nn.Linear(5, 128)
        self.layer2 = nn.Linear(128, 128)
        # self.layer3 = nn.Linear(128, 128)
        self.layer4 = nn.Linear(128, 3)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        # x = F.relu(self.layer3(x))
        return self.layer4(x)


class UnnormalizedInventory(Inventory):

    def __init__(self,
                 machines: int,
                 order_pipeline: GeometricOrderPipeline,
                 max_batch_size: int = 3,
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
        super().__init__(machines=machines, order_pipeline=order_pipeline, max_batch_size=max_batch_size,
                         mttf=mttf, a=a, ordering_cost=ordering_cost, emergency_cost=emergency_cost,
                         sorted_degradation=sorted_degradation)
    def __str__(self):
        return "Inventory-No Normalization"

    def copy(self):
        dummy = UnnormalizedInventory(
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

    def _get_obs(self, normalized=True):
        if self.sorted_degradation:
            array = np.append(np.sort(self.degradations),
                              [self.inventory_level,
                               *self.order_pipeline.pipeline])
        else:
            array = np.append(self.degradations,
                              [self.inventory_level,
                               *self.order_pipeline.pipeline])
        return array.astype(np.float32)


def probsp_rule(inventory:Inventory, state,n, xo):
    machines = inventory.num_machines
    higher_xo = np.sum(state[:-2] >= xo)
    d = n - state[-1] - state[-2] + higher_xo
    return max(d, 0)


def plot_dcl_probsp(n=None, xo=None):
	if n is None:
		n = 0
	if xo is None:
		xo = 55
	model = DQN2()
	dcl_model = torch.load(dcl_file, weights_only=False)
	new_dqn_model = model.state_dict()
	for dcl_key, dqn_key in zip(dcl_model.state_dict(), model.state_dict()):
		new_dqn_model[dqn_key] = dcl_model.state_dict()[dcl_key]
	model.load_state_dict(new_dqn_model)
	fig = plt.figure(dpi=dpi, figsize=fig_size, constrained_layout=True)
	fig.suptitle("DCLPro and ProBSP Ordering Decisions")

	subfigs = fig.subfigures(nrows=2, ncols=1)
	subfigs[0].suptitle("DCLPro Ordering Mechanism")
	subfigs[1].suptitle("ProBSP Ordering Mechanism")
	axs = subfigs[0].subplots(1, 3)
	x1 = np.arange(start=0., stop=99.5, step=5)
	x2 = x1.copy()
	ax = 0
	for inv in [0., 1.]:
		for out1 in [0., 1.]:
			if inv + out1 + 0 < 2:
				actions = np.ones((len(x1), len(x2))) * np.nan
				probsp_actions = np.ones((len(x1), len(x2))) * np.nan
				for i, deg1 in enumerate(x1):
					for j, deg2 in enumerate(x2):
						if deg2 >= deg1:
							obs = torch.tensor([[deg1, deg2, inv, out1, 0]], dtype=torch.float32)
							mask = [True, True, True] if inv + out1 + 0 == 0 else [True, True, False]
							forward_pass = model(obs)
							action = torch.argmax(torch.masked_fill(forward_pass, ~torch.tensor(mask), -1000))
							actions[j][i] = action.item()

				im = axs[ax].imshow(actions, cmap=cmap, norm=norm)
				im2 = axs[ax].imshow(probsp_actions, cmap=cmap, norm=norm)
				x_ticks = np.linspace(start=0, stop=len(x1) - 1, num=10, dtype=int)
				x_labels = ["{:.1f}".format(i / len(x1)) for i in x_ticks]
				axs[ax].set_xticks(x_ticks)
				axs[ax].set_xticklabels(x_labels)
				axs[ax].set_yticks(x_ticks)
				axs[ax].set_yticklabels(x_labels)
				axs[ax].invert_yaxis()
				axs[ax].set_xlabel("Lowest Degradation")
				axs[ax].set_ylabel("Highest Degradation")
				axs[ax].set_title(f"I={inv:.0f}, $O$={out1:.0f}")
				for i in range(len(x1)):
					for j in range(len(x2)):
						c = '0' if actions[i, j] in [1, 2] else '1'
						text = axs[ax].text(j, i, "{:.0f}".format(actions[i, j]),
						                    ha="center", va="center", color=c)

				for row in [0, 1]:
					axs[ax].text(12, 4, "Invalid States", ha="center", va="center", color="black",
					             fontsize=20, style='italic', bbox=dict(facecolor='lightgrey', alpha=0.5))

				ax += 1
				plt.draw()

	axs = subfigs[1].subplots(1, 3)
	x1 = np.arange(start=0., stop=99.5, step=5)
	x2 = x1.copy()
	ax = 0
	for inv in [0., 1.]:
		for out in [0., 1.]:
			if inv + out < 2:
				actions = np.ones((len(x1), len(x2))) * np.nan
				probsp_actions = np.ones((len(x1), len(x2))) * np.nan
				for i, deg1 in enumerate(x1):
					for j, deg2 in enumerate(x2):
						if deg2 >= deg1:
							actions[j][i] = probsp_rule(inventory, np.array([deg1, deg2, inv, out]), n, xo)

				im = axs[ax].imshow(actions, cmap=cmap, norm=norm)
				im2 = axs[ax].imshow(probsp_actions, cmap=cmap, norm=norm)
				x_ticks = np.linspace(start=0, stop=len(x1) - 1, num=10, dtype=int)
				x_labels = ["{:.1f}".format(i / len(x1)) for i in x_ticks]
				axs[ax].set_xticks(x_ticks)
				axs[ax].set_xticklabels(x_labels)
				axs[ax].set_yticks(x_ticks)
				axs[ax].set_yticklabels(x_labels)
				axs[ax].invert_yaxis()
				axs[ax].set_xlabel("Lowest Degradation")
				axs[ax].set_ylabel("Highest Degradation")
				axs[ax].set_title(f"I={inv:.0f} and O={out:.0f}")
				for i in range(len(x1)):
					for j in range(len(x2)):
						c = '0' if actions[i, j] in [1, 2] else '1'
						text = axs[ax].text(j, i, "{:.0f}".format(actions[i, j]),
						                    ha="center", va="center", color=c)

				for row in [0, 1]:
					axs[ax].text(12, 4, "Invalid States", ha="center", va="center", color="black",
					             fontsize=20, style='italic', bbox=dict(facecolor='lightgrey', alpha=0.5))

				ax += 1
				plt.draw()
	plt.draw()
	plt.show()


def evaluate_DCL():
	model = MaskedDQN(inventory, dqn_model=DQN2)
	dcl_model = torch.load(dcl_file, weights_only=False)
	new_dqn_model = model.policy_net.state_dict()
	for dcl_key, dqn_key in zip(dcl_model.state_dict(), model.policy_net.state_dict()):
		new_dqn_model[dqn_key] = dcl_model.state_dict()[dcl_key]
	model.policy_net.load_state_dict(new_dqn_model)
	evaluate_policy(inventory, policy=model)


dcl_file = 'dcl_data/l3_sens/2machines_mttf10_deter.pth'
MTTF=10
CE=5

fig_size = (18, 10)
dpi = 90
colors = ['black', 'silver', 'bisque']
cmap = ListedColormap(colors)


# Define bounds for the values
bounds = np.arange(1, 3)
norm = BoundaryNorm(bounds, cmap.N)
# order_pipeline = GeometricOrderPipeline(2, 1/3)
order_pipeline = DeterministicOrderPipeline(2, 3)
# order_pipeline = EmpiricalOrderPipeline(2, [1,2,3,4,5], [0.2,0.1,0.3,0.3,0.1])
# order_pipeline = UniformOrderPipeline(2, 1, 5)
inventory = UnnormalizedInventory(2, order_pipeline, sorted_degradation=True, mttf=MTTF, emergency_cost=CE)
regular_inventory = Inventory(2, order_pipeline, sorted_degradation=True, mttf=MTTF, emergency_cost=CE)

# Example
n, xo = 0, 32.5
plot_dcl_probsp(n=n, xo=xo)
# evaluate_DCL()  # Evaluate the performance of the imported DCL network
# evaluate_policy(regular_inventory, ProBSP(regular_inventory, n,xo,5))  # evaluate the performance of the ProBSP