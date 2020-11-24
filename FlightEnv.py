import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf

from functools import partial
from os import listdir, makedirs
from os.path import isfile, join, exists
import imageio


"""

FlightEnv provides the physics and an integrator. Derived classes are 
responsible for providing a reward function, a completion objective and
additional rendering.

"""


class FlightObject:
    id_ = 0
    def __init__(self):
        self.pos = np.array([0, 0])
        self.vel = np.array([0.0, 0])
        self.orient = np.array([1, 0])
        
        self.gravityExempt = False
        self.liftCoef = 1.0
        self.dragCoef = 1.0
        self.mass = 1.0
        
        # Accel is a flag telling us whether or not this object is accelerating (x, y)
        self.accel = [0, 0]
        
        # We apply a fixed acceleration, given by these values (x, y)
        self.accel_values = [2, 2]
        


class FlightEnv:
    def __init__(self, name = 'Nameless'):
        self.chaser = FlightObject()
        
        self.physics = {
            'g': .5,                   # acceleration of gravity
            'aoaCritical': np.pi/6,      # no lift beyond this angle
        }
        
        self.simulation = {
            't': 0,
            'timeStep': 1.0/120.0,
        }
        
        self.renderOpts = {
            'viewport': [5, 5],
            'arrowHeadLength': .1,
            'arrowHeadWidth': .1
        }
        
        # History
        self.history = {
            'positions': [],
            'actions': []
        }
        
        """ 
            Possible actions:
            
            # 0: no pitch, thrust on
            # 1: pitch up, thrust on
            # 2: pitch down, thrust on
            
            # 3: no pitch, thrust off
            # 4: pitch up, thrust off
            # 5: pitch down, thrust off 
        """
        self.action_space = list(range(6))
        
        self.chaserForces = lambda t, vel: sum(partial(FlightEnv.__calculateForces, self, self.chaser)(t, vel))
        
        # Used for saving plots (to an unused folder)
        self.name = name
        FlightObject.id_ = FlightObject.id_ + 1
        self.id_ = FlightObject.id_
        
    def Reset(self):
        pass
    
    def __liftModulator(self, y):
        mod = 1
        width = 1
        
        return mod*(1-np.tanh(y/width))
        
    # Obj is to be used only for static properties
    def __calculateForces(self, obj, t, vel):
        forces = []
        # Calculate gravity
        gravity = self.physics['g']*np.array([0, -1])
        if (obj.gravityExempt):
            gravity = 0*gravity
        forces.append(gravity)
        
        # Calculate thrust
        thrust = np.array([a*av/obj.mass for a, av in zip(obj.accel, obj.accel_values)])
        forces.append(thrust)
        
        # The rest of the forces depend on the velocity. Thus return if zero
        if (not np.linalg.norm(vel)):
            return forces
        
        # Calculate drag
        dragDir = -vel / np.linalg.norm(vel)
        drag = obj.dragCoef*np.dot(vel, vel)*dragDir
        
        # Calculate lift
        
        # Get angle of attack first
        phi_vel = np.arctan2(vel[1]/np.linalg.norm(vel), vel[0]/np.linalg.norm(vel))
        phi_ori = np.arctan2(obj.orient[1]/np.linalg.norm(obj.orient), obj.orient[0]/np.linalg.norm(obj.orient))
        angleOfAttack = phi_ori - phi_vel
        
        # If aoa negative or beyond critical, just set it to zero to kill the lift
        if angleOfAttack < 0 or angleOfAttack > self.physics['aoaCritical']:
            angleOfAttack = 0
            
        # Lift is perpendicular to velocity
        liftDir = np.array([-vel[1], vel[0]])/np.linalg.norm(vel)
        lift = self.__liftModulator(obj.pos[1])*obj.liftCoef*angleOfAttack*np.dot(vel, vel)*liftDir
        
        forces.append(drag)
        forces.append(lift)
        
        return forces
    
    def __calculateReward(self):
        return 1
    
    def __isSimulationFinished(self):
        return False
    
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
    def Step(self, action = 0):
        # Current time
        t = self.simulation['t']
        dt = self.simulation['timeStep']
        
        # Apply action
        self.chaser.accel[0] = True if action else False
        
        # Update chaser
        self.chaser.vel = self.__rk4_step(t, dt, self.chaser.vel, self.chaserForces)
        self.chaser.pos = self.chaser.pos + dt*self.chaser.vel
        
        # Update time
        self.simulation['t'] = t + dt
        
        # Save step to history
        self.history['positions'].append(self.chaser.pos)
        self.history['actions'].append(action)
        
        # Return an observation, which our policy will use to choose an action
        reward = self.__calculateReward()
        done = self.__isSimulationFinished()
        
        return [self.chaser.pos, self.chaser.vel, reward, done]

    def Render(self, save = False, debug = True):
        vp = self.renderOpts['viewport']
        plt.gca().set_xlim(self.chaser.pos[0]-vp[0]/2, self.chaser.pos[0] + vp[0]/2)
        plt.gca().set_ylim(self.chaser.pos[1]-vp[1]/2, self.chaser.pos[1] + vp[1]/2)
        
        
        # Draw chaser: position, orientation and velocity
        plt.plot(*self.chaser.pos, color='red', marker='o')
        plt.arrow(*self.chaser.pos, *(self.chaser.orient/np.linalg.norm(self.chaser.orient)), head_width=0, head_length=0)
        
        velnorm = np.linalg.norm(self.chaser.vel)
        chaservel = self.chaser.vel/velnorm if velnorm else self.chaser.vel
        plt.arrow(*self.chaser.pos, *chaservel, head_width=0, head_length=0, color = 'red')
        
        
        # Add debug text
        if (debug):
            velocityText = 'Velocity: ({:.3f}, {:.3f})'.format(self.chaser.vel[0], self.chaser.vel[1])
            plt.text(0.03, 0.95, velocityText, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
            
            forces = self.__calculateForces(self.chaser, self.simulation['t'], self.chaser.vel)
            lift = forces[-1]
            drag = forces[-2]
            
            liftText = 'Lift: ({:.3f}, {:.3f})'.format(lift[0], lift[1])
            plt.text(0.03, 0.90, liftText, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
            
            dragText = 'Drag: ({:.3f}, {:.3f})'.format(drag[0], drag[1])
            plt.text(0.03, 0.85, dragText, horizontalalignment='left', verticalalignment='center', transform=plt.gca().transAxes)
        
        if (save):
            directory = 'Plots/{}/{}/'.format(self.name, self.id_)
            if not exists(directory):
                makedirs(directory)

                
            filename = 'Plots/{}/{}/{:d}.png'.format(self.name, self.id_, round(1000*self.simulation['t']))
            plt.savefig(filename)
        else:
            plt.show()
            
        plt.clf()
            
    def SaveAnimation(self):
        path = 'Plots/{}/{}/'.format(self.name, self.id_)
        filenames = sorted([f for f in listdir(path) if isfile(join(path, f))], key = lambda x: int(x[:-4]))
        
        with imageio.get_writer(path + f"{self.id_}.gif", mode='I', duration = 1./20) as writer:    
            for filename in filenames:
                try:
                    image = imageio.imread(path + filename)
                    writer.append_data(image)
                except ValueError:
                    print(path+filename)
                    raise ValueError(path+filename)
        
        
    