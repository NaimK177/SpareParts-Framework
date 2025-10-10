# Degradation-aware Spare Parts management

This repository provides the complete implementation, data, and simulation environment used to study the integration 
of machine degradation information into spare parts ordering decisions under stochastic lead times and batch ordering 
constraints.

This repository focuses on two algorithms: DQN and PPO. 
The repository [DCL-Spare-Parts](https://github.com/NaimK177/DCL-Spare-Parts) on the other hands implement the DCL algorithm within our problem setting.

## Overview
Efficient spare parts management is essential for ensuring high availability and cost-effective maintenance 
in industrial systems.

This repository introduces a framework for learning degradation aware policies that learns to order spare parts 
in complex systems with **multiple machines**, **stochastic lead times**, and **batch ordering**.
In addition to learning DRL-based policies, this framework enables a simple implementation of heuristic policies
as shown in [Policies.py](Policies.py).

## Key Features
* A simple OpenAI Gym-based environment
* a general order pipeline enabling the selection of any lead time distribution via simple implementation in 
[NewEnvironment.py](NewEnvironment.py).
* A Masked DDQN algorithm 
* Tools to evaluate and analyze a DCL policy performance [DCLPolicyAnalysis.py](DCLPolicyAnalysis.py).

## Problem Description
The environment models a system of multiple machines subject to degradation following a gamma process.
When a machine fails, a spare part is required for replacement.
Orders for parts can be placed in batches, and the lead time before delivery is stochastic (can be also deterministic).
The objective is to minimize total long-run cost, consisting of:
* Holding cost for keeping spare parts in inventory,
* Ordering cost (fixed and variable components), and
* Emergency cost when a failure occurs and no spare is available.

The agent (policy) observes:
* The degradation states of all machines,
* The inventory level and outstanding orders,
And decides how many parts to order at each time step.

