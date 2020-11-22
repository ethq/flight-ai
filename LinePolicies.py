# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 11:48:01 2020

@author: Z
"""

# Policies for LineEnv
# LineEnv.step() returns [position, velocity, reward, done]


"""
If pos[0] > 0, releases thrust
If pos[1] < 0, applies thrust
"""
def ActOnUpOrDown(observation):
    pos, vel, reward, done = observation
    return True if pos[1] < 0 else False


"""
Always thrust.
"""
def AlwaysThrust(observation):
    pos, vel, reward, done = observation
    return True




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