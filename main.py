# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 10:44:48 2020

@author: Z
"""

"""
In this environment the goal is to keep the plane flying in a straight line.
"""

import numpy as np
import matplotlib.pyplot as plt

from LineEnv import LineEnv
import LinePolicies

env = LineEnv(name = 'start')
policy = LinePolicies.AlwaysThrust
policy = LinePolicies.ActOnUpOrDown

positions = []
actions = []
action = True
obs = []

for i in range(2000):
    obs = env.Step(action)
    action = policy(obs)
    
    actions.append(action)
    positions.append(env.chaser.pos)
    if (not i % 10):
        env.Render(True)
    
env.SaveAnimation()


# Draw trajectory
plt.plot(*list(zip(*positions)))
plt.show()

plt.plot(actions)
plt.show()