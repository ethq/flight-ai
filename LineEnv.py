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
        
        # Set the acceleration values to give a decent fit at full throttle for T = 2000.
        self.chaser.accel_values = [0.95, 0]
        
    def Render(self, save):
        # Figure out where we need to draw the line
        pos = self.chaser.pos
        vp = self.renderOpts['viewport']
        
        line = [[pos[0]-vp[0]/2, pos[0]+vp[0]/2], [self.y, self.y]]
        plt.plot(*line, color = 'green')
        
        
        super().Render(save)
        
    def __calculateReward(self):
        # Squared distance to the line
        d = (self.chaser.pos[1] - self.y)**2
        
        # I'll clamp it to some random value for now
        threshold = 10
        if (d > threshold):
            d = threshold
            
        return d