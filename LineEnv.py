# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 13:18:27 2020

@author: Z

"""

import numpy as np
import matplotlib.pyplot as plt

from FlightEnv import FlightEnv

class LineEnv(FlightEnv):
    def __init__(self, name):
        super().__init__(name)
        
        # For now, the line is straight and locked to y = 0
        self.y = 0
        
        # Set initial values to give OK performance with a simple policy
        self.chaser.vel = np.array([1.2, 0])
        self.chaser.orient = np.array([0.7, 0.14])
        self.chaser.liftCoef = 5.8
        self.chaser.dragCoef = 0.6
        self.chaser.accel[0] = 1
        self.chaser.accel_values = [0.95, 0]
        
    def Reset(self):
        super().Reset()
        self.__init__(self.name)
        
        return np.concatenate((self.chaser.pos, self.chaser.vel))
    
    def Render(self, save):
        # Figure out where we need to draw the line
        pos = self.chaser.pos
        vp = self.renderOpts['viewport']
        
        line = [[pos[0]-vp[0]/2, pos[0]+vp[0]/2], [self.y, self.y]]
        plt.plot(*line, color = 'green')
        
        
        super().Render(save)
        
    def CalculateReward(self, pos):
        # Multiplier, we don't want numbers too small
        mul = 120 # Initial value ~50
        
        # Squared distance to the line
        return -np.exp(mul*np.abs(pos[1])**2)