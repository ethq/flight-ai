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

#from FlightEnv import FlightEnv
from LineEnv import LineEnv

env = LineEnv(name = 'start')

positions = []
for i in range(1000):
    env.step(1)
    positions.append(env.chaser.pos)
    
    if (not i % 10):
        env.render(True)
    
env.save_animation()
