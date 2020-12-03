# -*- coding: utf-8 -*-
"""
In this environment the goal is to keep the plane flying in a straight line.
"""

import numpy as np
import matplotlib.pyplot as plt

from LineEnv import LineEnv
import LinePolicies

def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.empty(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards*discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(all_rewards, discount_rate) for reward in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    
    return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]


env = LineEnv(name = 'start')
policy = LinePolicies.AlwaysThrust
policy = LinePolicies.ActOnUpOrDownModulated
policy = LinePolicies.ThrustInRange
policy = LinePolicies.ThrustOnNegativeVelocityAndRange
policy = LinePolicies.VelocityAndPositionRange

positions = []
actions = []
rewards = []

action = True
obs = []

for i in range(2000):
    obs = env.Step(action)
    action = policy(obs)
    
    actions.append(action)
    rewards.append(obs[2])
#     if (not i % 50):
#         env.Render(True)
    
# env.SaveAnimation()

# How did we do?
#print(env.Score())
print(f'Total reward: {sum(rewards)}') 


# Draw trajectory
pos = list(zip(*env.history['positions']))
plt.plot(pos[1])
plt.title(f'Score: {env.Score()}\nPolicy: {policy.__name__}')

actions = np.array(actions) * max(pos[1])
plt.plot(actions, alpha=.3, color = 'black')
plt.show()