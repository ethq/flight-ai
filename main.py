# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 10:44:48 2020

@author: Z
"""

"""
In this environment the goal is to keep the plane flying in a straight line.
"""

#from FlightEnv import FlightEnv
from LineEnv import LineEnv

env = LineEnv(name = 'start')

for i in range(100):
    env.step(1)
    env.render(True)
    
env.save_animation()
