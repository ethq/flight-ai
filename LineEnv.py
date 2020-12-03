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
        
        # Return an observation, which our policy will use to choose an action
        reward = self._calculateReward()
        done = False
        
        return [self.chaser.pos, self.chaser.vel, reward, done]
    
    # Careful with this. May lead to messy design; we want separation of concerns and environment should primarily be focused on running
    def Score(self, metric = 0):
        if not metric:
            return sum([p[1]**2 for p in self.history['positions']])
        elif metric == 1:
            return 0
        
        return 0
    
    def Render(self, save):
        # Figure out where we need to draw the line
        pos = self.chaser.pos
        vp = self.renderOpts['viewport']
        
        line = [[pos[0]-vp[0]/2, pos[0]+vp[0]/2], [self.y, self.y]]
        plt.plot(*line, color = 'green')
        
        
        super().Render(save)
        
    def _calculateReward(self):
        # # For now, reward for going closer to 1, punishment otherwise
        # assert(len(self.history['positions']) > 1)
        # lastpos = self.history['positions'][-2]
        # if (abs(self.chaser.pos[1]) < abs(lastpos[1])):
        #     return 1
        # return -1
        
        # Multiplier, we don't want numbers too small
        mul = 1000
        
        # Squared distance to the line
        d = -mul*np.abs(self.chaser.pos[1])**2
            
        return d