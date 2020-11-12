import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class FlightEnv:
    def __init__(self):
        # Target position
        target = {}
        target.pos = np.array([5, 0])
        target.vel = np.array([0.0, 0])
        target.orient = np.array([1, 0])
        target.gravityExempt = True
        
        chaser = {}
        chaser.pos = np.array([0,0])
        chaser.vel = np.array([0.1,0])
        chaser.orient = np.array([1,0])
        chaser.liftCoef = 1.0
        chaser.dragCoef = 1.0
        
        physics = {}
        physics.g = 9.81
        
        self.target = target
        self.chaser = chaser
        self.physics = physics
        
    def calcForces(self, obj):
        # Calculate velocity-independent forces first
        gravity = g*np.array([0, -1])
        if (obj.gravityExempt):
            gravity = 0*gravity
            
        forceSet1 = gravity
        
        # If velocity is zero, just return the other forces
        if (not np.linalg.norm(obj.vel)):
            return forceSet1
        
        dragDir = -obj.vel / np.linalg.norm(obj.vel)
        drag = obj.dragCoef*np.dot(obj.vel, obj.vel)*dragDir
        
        angleOfAttack = 0
        np.dot(obj.vel, obj.orient)/(np.linalg.norm(obj.orient)*np.linalg.norm(obj.vel))
            
        # Rotation by 90 deg ccw, assumes velocities given in cartesian coords
        liftDir = [-obj.vel[1], obj.vel[0]] 
        lift = obj.liftCoef*angleOfAttack*np.dot(obj.vel, obj.vel)*liftDir
        
        forceSet2 = drag + lift
        
        return forceSet1 + forceSet2
    
    


    # Evolves y(t) from t0 to t with stepsize dt.
    # f(t,y) should evaluate dy/dt
    def rk4(self, t0, t, y0, dt, f): 
        # Count number of iterations
        n = (int)((t - t0)/dt)  
        
        y = np.zeros(n)
        y[0] = y0
        
        # Iterate for number of iterations 
        for i in range(1, n + 1): 
            y[i] = self.rk4_step(t0, dt, y[i-1], f)
            k1 = dt * f(t0, y[i-1]) 
            k2 = dt * f(t0 + 0.5 * dt, y[i-1] + 0.5 * k1) 
            k3 = dt * f(t0 + 0.5 * dt, y[i-1] + 0.5 * k2) 
            k4 = dt * f(t0 + dt, y[i-1] + k3) 
      
            # Update next value of y 
            y[i] = y[i-1] + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4) 
      
            # Update next value of t
            t0 = t0 + dt
        return y
    
    def rk4_step(self, t0, dt, y, f):
        k1 = dt * f(t0, y) 
        k2 = dt * f(t0 + 0.5 * dt, y + 0.5 * k1) 
        k3 = dt * f(t0 + 0.5 * dt, y + 0.5 * k2) 
        k4 = dt * f(t0 + dt, y + k3) 
        
        return y + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4) 