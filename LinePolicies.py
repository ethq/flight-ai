# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 11:48:01 2020

@author: Z
"""

import numpy as np
import tensorflow as tf

# Policies for LineEnv
# LineEnv.step() returns [position, velocity, reward, done]



def ActOnUpOrDown(observation):
    pos, vel, reward, done = observation
    return True if pos[1] < 0 else False

def ActOnUpOrDownModulated(obs):
    pos, vel, reward, done = obs
    return True if pos[1] < .02 else False

def ThrustInRange(obs):
    pos, vel, reward, done = obs
    y = pos[1]
    
    if y > -0.1 and y < 0.01:
        return True
    else:
        return False
    
def ThrustOnNegativeVelocity(obs):
    p, v, r, d = obs
    
    if v[1] < 0:
        return True
    return False

def ThrustOnNegativeVelocityAndRange(obs):
    p, v, r, d = obs
    
    if -v[1] > 0.005:
        return True
    return False

def VelocityAndPositionRange(obs):
    p,v,r,d = obs
    print(r)
    # Relax plane to y = 0 as quickly as possible
    
    if p[1] > 0.025:
        return False
    elif -v[1] > 0.0008:
        return True
    return False

def AlwaysThrust(observation):
    pos, vel, reward, done = observation
    return True

def SimpleNet(obs):
    p,v,r,d = obs
    
    # Architecture
    n_inputs = 4
    n_hidden = 4
    n_outputs = 1
    initializer = tf.contrib.layers.variance_scaling_initializer()
    
    # Building
    X = tf.placeholder(tf.float32, shape=[None, n_inputs])
    hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
    logits = tf.layerse.dense(hidden, n_outputs, kernel_initializer=initializer)
    outputs = tf.nn.sigmoid(logits)
    
    # Sample multinomial distrib to pick action given probabilities
    p_thrust_nothrust = tf.concat(axis=1, values=[outputs, 1-outputs])
    action = tf.multinomial(tf.log(p_thrust_nothrust), num_samples=1)
    
    init = tf.global_variables_initializer()






def TestActOnUpOrDown():
    success = 0
    total = 2
    
    pos_up = [0, 5]
    thrust_off = ActOnUpOrDown(pos_up, [0, 0], True, True)
    success += 1 if not thrust_off else 0
    
    pos_down = [0, -5]
    thrust_on = ActOnUpOrDown(pos_down, [0, 0], True, True)
    success += 1 if thrust_on else 0
    
    print('Passed {}/{} tests.'.format(success, total))
    
    
if __name__ == '__main__':
    TestActOnUpOrDown()