# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 09:42:53 2015

@author: zwang04
"""

import numpy as np
from numpy import dot, sum, tile, linalg 
from numpy.linalg import inv 
import matplotlib.pyplot as plt
from numpy import *
from numpy.linalg import inv
import firstOrderEKF as ekf

data = np.load('pendulum.npz') 
obs = data['observations']
etat = data['states']
temp = data['time']
x0 = data['x0']
r = data['r']
q = data['q']
p0 = data['p0']
g = data['g']
dt = data['dt']
angMeasure = data['anglemeasurements']

np.delete(temp, 1, 0)

plt.figure(1)
plt.plot(angMeasure ,'ro')

def transition_function(state):
    return np.sin(state)+r
    
def transition_function_jacobian(state):
    return np.array([1,dt],[-g*np.sin(state)*dt,1]) 

# Q
transition_covariance = np.array([[q*(dt**3)/3,q*(dt**2)/2],[q*(dt**2)/2,q*dt]]) 
#current_state_covariance = 

#def observation_function(state):
#def observation_function_jacobian(state):
# transition_covariance
# current_state_mean
# current_state_covariance
N_iter = 50
for i in arange(0, N_iter):
    (predicted_state_mean, predicted_state_covariance) = filter_predict(transition_function, transition_function_jacobian, transition_covariance, x0, current_state_covariance)    

plt.figure(2)
plt.plot(predicted_state_mean ,'ro')