#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : grid_search.py
# description     : use the Pytorch agent and to explore the different parameter values in terms of performance
# author          : Nicolas Coucke
# date            : 2022-08-29
# version         : 1
# usage           : grid_search.py
# notes           : install the packages with "pip install -r requirements.txt"
# python_version  : 3.9.2
# ==============================================================================



from utils import symmetric_matrix, eucl_distance
from environment import Environment
from agent_RL import Gina, Guido

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
import tkinter as tk
import random

import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# define variables for environment
fs = 30 # Hertz
duration = 30 # Seconds
stimulus_position = [0, 0] # m, m
stimulus_decay_rate = 0.02 # in the environment
stimulus_scale = 10 # in the environment
stimulus_sensitivity = 1 # of the agent
starting_position = [0, -100] 
starting_orientation = 0 
movement_speed = 10
delta_orientation = 0.1*np.pi # turning speed
agent_radius = 2
agent_eye_angle = 45

f_sens = 1.5
f_motor = 1.0
a_sens = 0.5
a_con = 5
a_ips = 3
a_motor = 1

frequency = np.array([f_sens, f_motor])
phase_coupling = np.array([a_sens, a_con, a_ips, a_motor])
k = 5

env = Environment(fs, duration, stimulus_position, stimulus_decay_rate,
     stimulus_scale, stimulus_sensitivity, starting_position, starting_orientation, movement_speed, agent_radius, agent_eye_angle, delta_orientation)

# create an agent as policy 
#policy = Gina(device).to(device)
policy = Guido(device, fs, frequency, phase_coupling, k).to(device)


n_episodes = 10
# do ten episodes for each parameter combination

fig, (ax1, ax2) = plt.subplots(1,2)

for i in range(n_episodes):

    starting_orientation = random.uniform(0, 2*np.pi)

    starting_orientation = 0
    # start on a random point on the line going from 10 to 100m from center
    starting_position = np.array([0, -random.randrange(95, 105)])

    # the environment keeps track of the agent's position
    state = env.reset(starting_position, starting_orientation)
    policy.set_phases(torch.tensor([0., 0., 0., 0.]))

    # Complete the whole episode
    for t in range(duration * fs):
        action, log_prob = policy.act(state)
        state, reward, done = env.step(action, 10)
        if done:
            break 

    # plot the trajectory of the agant in the environment
    i = 0
    x_prev = env.position_x[0]
    y_prev = env.position_y[0]
    for x, y in zip(env.position_x, env.position_y):
        # later samples are more visible
        a = 0.5*i/len(env.position_x)
        ax1.plot([x_prev, x], [y_prev, y], alpha = a, color = 'red')
        x_prev = x
        y_prev = y
        i+=1
    ax1.set_xlim([-150, 150])
    ax1.set_ylim([-150, 150])




# plot the environment with stimulus concentration
N = 1000
x = np.linspace(-150, 150, N)
y = np.linspace(-150, 150, N)
xx, yy = np.meshgrid(x, y)
xx, yy = np.meshgrid(x, y)
zz = np.sqrt(xx**2 + yy**2)   
zs = stimulus_scale * np.exp( - stimulus_decay_rate * zz)
environment_plot = ax1.contourf(x, y, zs)
ax1.axis('scaled')
plt.colorbar(environment_plot, ax=ax1)

def draw_curve_horizontal(p1, p2):
   x = np.linspace(p1[0], p2[0], 100)

   height_difference = np.linspace(p1[1], p2[1], 100)

   y =  0.2*np.sin(np.pi *(x - p1[0])/2) + height_difference

   return x, y

def draw_curve_vertical(p1, p2):
   y = np.linspace(p1[1], p2[1], 100)

   height_difference = np.linspace(p1[0], p2[0], 100)

   x =  0.2*np.sin(np.pi *(y - p1[1])/2) + height_difference

   return x, y

def draw_anti_curve_horizontal(p1, p2):
   x = np.linspace(p1[0], p2[0], 100)

   height_difference = np.linspace(p1[1], p2[1], 100)

   y = - 0.2*np.sin(np.pi *(x - p1[0])/2) + height_difference

   return x, y

def draw_anti_curve_vertical(p1, p2):
   y = np.linspace(p1[1], p2[1], 100)

   height_difference = np.linspace(p1[0], p2[0], 100)

   x = - 0.2*np.sin(np.pi *(y - p1[1])/2) + height_difference

   return x, y

# visualize the connection strengths
# phase difference between oscillators
ax2.set_xlim([-1.5, 1.5])
ax2.set_ylim([-1.5, 1.5])
pos1 = [-1, 1]
pos2 = [1, 1.001]
pos3 = [-1.001, -1]
pos4 = [1.001, -1.001]

oscillator1 = ax2.scatter(-1,1, s = f_sens * 200, color = 'grey') 
oscillator2 = ax2.scatter(1,1, s = f_sens * 200, color = 'grey')
oscillator3 = ax2.scatter(-1,-1, s = f_motor * 200, color = 'grey')
oscillator4 = ax2.scatter(1,-1, s = f_motor * 200, color = 'grey')

# in-phase curves
# sensor and motor connections
x, y = draw_curve_horizontal(pos1, pos2)
line_1_2, = ax2.plot(x, y, color = 'blue', linewidth = a_sens)
x, y = draw_anti_curve_horizontal(pos3, pos4)
line_3_4, = ax2.plot(x, y, color = 'blue', linewidth = a_motor)

# ipsilateral connections
x, y = draw_curve_vertical(pos1, pos3)
line_1_3, = ax2.plot(x, y, color = 'blue', linewidth = a_ips)
x, y = draw_anti_curve_vertical(pos2, pos4)
line_2_4, = ax2.plot(x, y, color = 'blue', linewidth = a_ips)

# contralateral connections
x, y = draw_curve_horizontal(pos1, pos4)
line_1_4, = ax2.plot(x, y, color = 'blue', linewidth = a_con)
x, y = draw_anti_curve_horizontal(pos2, pos3)
line_2_3, = ax2.plot(x, y, color = 'blue', linewidth = a_con)

# anti-phase curves
# sensor and motor connections
x, y = draw_anti_curve_horizontal(pos1, pos2)
line_1_2, = ax2.plot(x, y, color = 'red', linewidth = a_sens / k)
x, y = draw_curve_horizontal(pos3, pos4)
line_3_4, = ax2.plot(x, y, color = 'red', linewidth = a_motor / k)

# ipsilateral connections
x, y = draw_anti_curve_vertical(pos1, pos3)
line_1_3, = ax2.plot(x, y, color = 'red', linewidth = a_ips / k)
x, y = draw_curve_vertical(pos2, pos4)
line_2_4, = ax2.plot(x, y, color = 'red', linewidth = a_ips / k)

# contralateral connections
x, y = draw_anti_curve_horizontal(pos1, pos4)
line_1_4, = ax2.plot(x, y, color = 'red', linewidth = a_con / k)
x, y = draw_curve_horizontal(pos2, pos3)
line_2_3, = ax2.plot(x, y, color = 'red', linewidth = a_con / k)

ax2.axis('scaled')
plt.show()
