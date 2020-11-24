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
policy = LinePolicies.ActOnUpOrDownModulated
policy = LinePolicies.ThrustInRange
policy = LinePolicies.ThrustOnNegativeVelocityAndRange
policy = LinePolicies.VelocityAndPositionRange

positions = []
actions = []
action = True
obs = []

for i in range(4000):
    obs = env.Step(action)
    action = policy(obs)
    
#     if (not i % 50):
#         env.Render(True)
    
# env.SaveAnimation()

# How did we do?
print(env.Score())


# Draw trajectory
pos = list(zip(*env.history['positions']))
plt.plot(pos[1])
plt.title(f'Score: {env.Score()}\nPolicy: {policy}')

actions = np.array(env.history['actions'])
actions = actions * max(pos[1])

plt.plot(actions)
plt.show()