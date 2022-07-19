#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : single_agent_visualization.py
# description     : Simulate an agent that moves towards a stimulus in the environment
#                   with real-time animation
# author          : Nicolas Coucke
# date            : 2022-07-19
# version         : 1
# usage           : python single_agent_visualization
# notes           : install the packages with "pip install -r requirements.txt"
# python_version  : 3.9.2
# ==============================================================================

import matplotlib.pyplot as plt
import torch
import numpy as np
from utils import symmetric_matrix, eucl_distance
from agent import Agent
import time
from matplotlib import animation
import tkinter as tk


# define variables of simulated environment
fs = torch.as_tensor([100])
duration = torch.as_tensor([50]) # s
stimulus_position = torch.tensor([-75., 75.]) # [m, m]
stimulus_gradient = torch.as_tensor([5])
max_stimulus_value = torch.as_tensor([300.])
periodic_randomization = False

# instantiate an agent
agent_id = 1
stimulus_sensitivity = torch.as_tensor([5.])
phase_coupling_matrix = symmetric_matrix(5, 4)
anti_phase_coupling_matrix = symmetric_matrix(1, 4)
initial_phases = torch.tensor([0., torch.pi, 0.6, torch.pi]) # rad
frequencies = torch.tensor([1, 1, 1, 1 ]) # Hertz
movement_speed = torch.as_tensor([20]) # m/s
agent_position = torch.tensor([0., 0.]) # [m, m]
agent_orientation = torch.as_tensor([0.]) # rad

# create agent object
agent = Agent(agent_id, stimulus_sensitivity, phase_coupling_matrix, anti_phase_coupling_matrix,
            initial_phases, frequencies, movement_speed)


# generate time and initialize arrays to store trajectory
t = torch.linspace(start=0, end=float(duration), steps=int(duration * fs))
num_frames = int(duration * fs)
agent_position_x = []
agent_position_y = []
oscillator_spaces_x = np.zeros((4,duration*fs)) # for plotting phase spaces
oscillator_spaces_y = np.zeros((4,duration*fs))


# create objects for visualization
fig, (ax1, ax2, ax3) = plt.subplots(1,3)

# agent trajectory 
ax1.scatter(stimulus_position[0], stimulus_position[1])
ax1.set_xlim([-100,100])
ax1.set_ylim([-100,100])
ax1.set_title('Agent Trajectory')
line, = ax1.plot(0,0)

# phase difference between oscillators
ax2.set_xlim([-1.5, 1.5])
ax2.set_ylim([-1.5, 1.5])
oscillator1 = ax2.scatter(-1,1, s = 20) 
oscillator2 = ax2.scatter(1,1, s = 20)
oscillator3 = ax2.scatter(-1,-1, s = 20)
oscillator4 = ax2.scatter(-1,-1, s = 20)
oscillators = [oscillator1, oscillator2, oscillator3, oscillator4]
line_1_2, = ax2.plot([-1, 1], [1, 1])
line_1_3, = ax2.plot([-1, -1], [1, -1])
line_1_4, = ax2.plot([-1, 1], [1, -1])
line_2_3, = ax2.plot([1, -1], [1, -1])
line_2_4, = ax2.plot([1, 1], [1, -1])
line_3_4, = ax2.plot([-1, 1], [-1, -1])
lines = [line_1_2, line_1_3, line_1_4, line_2_3, line_2_4, line_3_4]
line_oscillators = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
ax2.set_title('Phase difference between oscillators')

# phase space of oscillators
ax3.set_xlim([-np.pi, np.pi])
ax3.set_ylim([-2, 6])
oscillator_space_1, = ax3.plot([], [], alpha=0.3)
oscillator_space_2, = ax3.plot([], [], alpha=0.3)
oscillator_space_3, = ax3.plot([], [], alpha=0.3)
oscillator_space_4, = ax3.plot([], [], alpha=0.3)
oscillator_spaces = [oscillator_space_1, oscillator_space_2, oscillator_space_3, oscillator_space_4]
ax3.set_title('Phase difference oscillators')


def update_simulation(i):
    """
    Function that is called by FuncAnimation at every timestep
    updates the simulation by one timestep and updates the plots correspondingly
    """
    # get the current eye positions of the agent
    left_eye_position, right_eye_position = agent.eye_positions()

    # calculate the stimulus intensity at the eye positions
    stimulus_intensity_left = max_stimulus_value - stimulus_gradient * eucl_distance(stimulus_position, left_eye_position)
    stimulus_intensity_right = max_stimulus_value - stimulus_gradient * eucl_distance(stimulus_position, right_eye_position)

    # get agent movement based on stimulus intensities
    agent_position, agent_orientation, phases, phase_differences  = agent.next_timestep(i, stimulus_intensity_left, stimulus_intensity_right, periodic_randomization)
    
    # set the thickness of the edges to the phase differences between oscillators
    for L in range(6):
        between_oscillator = np.abs(phases[line_oscillators[L][0]-1] - phases[line_oscillators[L][1]-1]) % 2 * torch.pi
        lines[L].set_linewidth(between_oscillator )

    # plot the trajectory of the individual oscillators in phase space
    for S in range(4):
        oscillator_spaces_x[S,i] = phases[S]
        oscillator_spaces_y[S,i] = phase_differences[S]
        oscillator_spaces[S].set_xdata(oscillator_spaces_x[S,:i])
        oscillator_spaces[S].set_ydata(oscillator_spaces_y[S,:i])

    # plot the agent position in the environment
    agent_position_x.append(agent_position[0])
    agent_position_y.append(agent_position[1])
    line.set_xdata(agent_position_x)
    line.set_ydata(agent_position_y)

    return line, line_1_2, line_1_3, line_1_4, line_2_3, line_2_4, line_3_4, oscillator_space_1, oscillator_space_2, oscillator_space_3, oscillator_space_4

anim = animation.FuncAnimation(fig, update_simulation, frames = range(duration * fs), interval = 1,
        blit = True)
plt.show()
#anim.save('linechart.gif')


