import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf

from functools import partial
from os import listdir, makedirs
from os.path import isfile, join, exists
import imageio

class FlightObject:
    id_ = 0
    def __init__(self):
        self.pos = np.array([0, 0])
        self.vel = np.array([0, 0])
        self.orient = np.array([1, 0])
        
        self.gravityExempt = False
        self.liftCoef = 1.0
        self.dragCoef = 1.0
        self.mass = 1.0
        


class FlightEnv:
    def __init__(self, name = 'Nameless'):
        chaser = FlightObject()
        chaser.pos = np.array([0,0])
        chaser.vel = np.array([0.0,0])
        chaser.orient = np.array([1,0])
        chaser.liftCoef = 1.0
        chaser.dragCoef = 1.0
        
        self.physics = {
            'g': 9.81
        }
        
        self.simulation = {
            't': 0,
            'timeStep': 1.0/60.0
        }
        
        self.target = target
        self.chaser = chaser
        
        # 0: thrust on
        # 1: thrust off
        # 2: no pitch
        # 3: pitch up
        # 4: pitch down
        self.action_space = [0, 1, 2]
        
        self.chaserForces = partial(FlightEnv.__calcForces, self, self.chaser)
        
        # Used for saving to an unused folder
        self.name = name
        FlightObject.id_ = FlightObject.id_ + 1
        self.id_ = FlightObject.id_
        
    # Obj is to be used only for static properties
    def __calcForces(self, obj, t, vel):
        # Calculate velocity-independent forces first
        gravity = self.physics['g']*np.array([0, -1])
        if (obj.gravityExempt):
            gravity = 0*gravity
            
        forceSet1 = gravity
        
        # If velocity is zero, just return the other forces
        if (not np.linalg.norm(vel)):
            return forceSet1
        
        dragDir = -vel / np.linalg.norm(vel)
        drag = obj.dragCoef*np.dot(vel, vel)*dragDir
        
        angleOfAttack = np.dot(vel, obj.orient)/(np.linalg.norm(obj.orient)*np.linalg.norm(vel))
            
        # Rotation by 90 deg ccw, assumes velocities given in cartesian coords
        liftDir = np.array([-vel[1], vel[0]])
        lift = obj.liftCoef*angleOfAttack*np.dot(vel, vel)*liftDir
        
        forceSet2 = drag + lift
        
        return forceSet1 + forceSet2
    
    # Might need adaptive version (?)
    # f should return the rhs of the ODE dv/dt = 1/m*f(t, v)
    # we assume it to be vectorized
    def __rk4_step(self, t, dt, v, f):        
        k1 = dt*f(t, v)
        k2 = dt*f(t + dt/2, v + k1/2)
        k3 = dt*f(t + dt/2, v + k2/2)
        k4 = dt*f(t + dt, v + k3)
        
        return v + (1.0 / 6.0)*(k1 + 2*k2 + 2*k3 + k4) 
    
    
    ## Public methods ##
    
    # Steps the simulation and performs the action
    def step(self, action=0):
        # Current time
        t = self.simulation['t']
        dt = self.simulation['timeStep']
        
        # Apply action
        
        
        # Update target
        self.target.vel = self.__rk4_step(t, dt, self.target.vel, self.targetForces)
        self.target.pos = self.target.pos + dt*self.target.vel
        
        # Update chaser
        self.chaser.vel = self.__rk4_step(t, dt, self.chaser.vel, self.chaserForces)
        self.chaser.pos = self.chaser.pos + dt*self.chaser.vel
        
        # Update time
        self.simulation['t'] = t + dt
        
        # Return an observation, which our policy will use to choose an action
        reward = 1
        done = False
        
        return [self.chaser.pos, self.chaser.vel, self.target.pos, self.target.vel, reward, done]

    def render(self, save = False): 
        plt.clf()
        plt.gca().set_xlim(-10, 10)
        plt.gca().set_ylim(-10, 10)
        
        # Draw chaser: position, orientation and velocity
        plt.plot(*self.chaser.pos, color='red', marker='o')
        plt.arrow(*self.chaser.pos, *(self.chaser.orient), head_width=.3, head_length=.3)
        
        velnorm = np.linalg.norm(self.chaser.vel)
        chaservel = self.chaser.vel/velnorm if velnorm else self.chaser.vel
        plt.arrow(*self.chaser.pos, *chaservel, head_width=.3, head_length=.3)
        
        
        
        # Draw target: position, orientation and velocity
        plt.plot(*self.target.pos, color='blue', marker='x')
        
        
        if (save):
            directory = 'Plots/{}/{}/'.format(self.name, self.id_)
            if not exists(directory):
                makedirs(directory)

                
            filename = 'Plots/{}/{}/{:.3f}.png'.format(self.name, self.id_, self.simulation['t'])
            plt.savefig(filename)
        else:
            plt.show()
            
    def save_animation(self):
        path = 'Plots/{}/{}/'.format(self.name, self.id_)
        
        with imageio.get_writer(path + f"{self.id_}.gif", mode='I', duration = 1./60) as writer:
            filenames = [f for f in listdir(path) if isfile(join(path, f))]
            for filename in filenames:
                image = imageio.imread(path + filename)
                writer.append_data(image)
        
        
    