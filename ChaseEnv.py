# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 13:18:27 2020

@author: Z
"""

"""

Objective: move behind target, axis-aligned.

"""

from FlightEnv import FlightEnv, FlightObject

class MoveToTargetEnv(FlightEnv):
    def __init__(self, name):
        super().__init__(name)
        
        # Target properties
        target = FlightObject()
        target.pos = np.array([5, 0])
        target.vel = np.array([0.0, 0])
        target.orient = np.array([1, 0])
        target.gravityExempt = True
        
        self.targetForces = partial(FlightEnv.__calcForces, self, self.target)