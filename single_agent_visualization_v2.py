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
import cmath


# define variables of simulated environment
fs = torch.as_tensor([50])
duration = torch.as_tensor([40]) # s
stimulus_position = torch.tensor([0., 0.]) # [m, m]
stimulus_decay_rate = torch.as_tensor([0.2])
stimulus_scale = torch.as_tensor([300.])
periodic_randomization = False

# instantiate an agent
agent_id = 1
stimulus_sensitivity = torch.as_tensor([20])
phase_coupling_matrix = symmetric_matrix(0.05, 4)
phase_coupling_matrix = torch.tensor([[0.00, 0.01, 0.05, 0.01],
                                      [0.01, 0.00, 0.01, 0.05],
                                      [0.05, 0.01, 0.00, 0.01],
                                      [0.01, 0.05, 0.01, 0.00]])
anti_phase_coupling_matrix = symmetric_matrix(0.001, 4)
initial_phases = torch.tensor([0., 0, 0.1, 0.1*torch.pi]) # rad
frequencies = torch.tensor([10, 10, 10, 10 ]) # Hertz
movement_speed = torch.as_tensor([8]) # m/s
agent_position = torch.tensor([0., -50.]) # [m, m]
agent_orientation = torch.as_tensor([0.]) # rad

# create agent object
agent = Agent(agent_id, stimulus_sensitivity, phase_coupling_matrix, anti_phase_coupling_matrix,
            initial_phases, frequencies, movement_speed, agent_position, agent_orientation)


# generate time and initialize arrays to store trajectory
t = torch.linspace(start=0, end=float(duration), steps=int(duration * fs))
num_frames = int(duration * fs)
agent_position_x = []
agent_position_y = []
oscillator_spaces_x = np.zeros((4,duration*fs)) # for plotting phase spaces
oscillator_spaces_y = np.zeros((4,duration*fs))

sensory_phase_difference = []
sensor_left_input = []
sensor_right_input = []
motor_phase_difference = []

# create objects for visualization
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4)
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
ax3.set_xlim([0, 2 * np.pi])
ax3.set_ylim([-20, 20])
oscillator_space_1 = ax3.scatter(0, 0, alpha=0.2, edgecolors=None, linewidths = 0, s = 1)
oscillator_space_2 = ax3.scatter(0, 0, alpha=0.2, edgecolors=None, linewidths = 0, s = 1)
oscillator_spaces = [oscillator_space_1, oscillator_space_2]
ax3.set_title('Phase difference oscillators')

ax4.set_xlim([0, fs * duration])
ax4.set_ylim([-2, 2*np.pi])
sensory_phase_line, = ax4.plot(0,0)
sensor_left_line, = ax4.plot(0,0)
sensor_right_line, = ax4.plot(0,0)
motor_phase_line, = ax4.plot(0,0)

ax4.legend([sensory_phase_line, sensor_left_line, sensor_right_line, motor_phase_line], ["sensory phase", "left sensor", "right sensor", "motor phase"])
def get_stimulus_indensity(distance):
    intensity = stimulus_scale * np.exp( - stimulus_decay_rate * distance )



def update_simulation(i):
    """
    Function that is called by FuncAnimation at every timestep
    updates the simulation by one timestep and updates the plots correspondingly
    """
    # get the current eye positions of the agent
    left_eye_position, right_eye_position = agent.eye_positions()

    # calculate the stimulus intensity at the eye positions
    left_eye_distance = eucl_distance(stimulus_position, left_eye_position)
    right_eye_distance = eucl_distance(stimulus_position, right_eye_position)
    print( - stimulus_decay_rate * left_eye_distance)
    stimulus_intensity_left = stimulus_scale * np.exp( - stimulus_decay_rate * left_eye_distance)
    stimulus_intensity_right = stimulus_scale * np.exp( - stimulus_decay_rate * right_eye_distance)

    
    previous_phases = agent.phases
    # get agent movement based on stimulus intensities
    agent_position, agent_orientation, phases, phase_differences  = agent.next_timestep(i, stimulus_intensity_left, stimulus_intensity_right, periodic_randomization)
    
    # set the thickness of the edges to the phase differences between oscillators
    for L in range(6):
        between_oscillator = np.abs(phases[line_oscillators[L][0]-1] - phases[line_oscillators[L][1]-1]) % 2 * torch.pi
        lines[L].set_linewidth(between_oscillator )

    # plot the trajectory of the individual oscillators in phase space
    # between sensory oscillators
    oscillator_spaces_x[0,i] = (phases[0] - phases[1]) % 2 * np.pi # absolute phase modulo 2pi
    oscillator_spaces_y[0,i] = (phases[0] - phases[1]) - (previous_phases[0] - previous_phases[1]) * fs # phase change
    #oscillator_spaces[0].set_xdata(oscillator_spaces_x[0,:])
    #oscillator_spaces[0].set_ydata(oscillator_spaces_y[0,:])
    plotdata = np.transpose(np.asarray((oscillator_spaces_x[0,:i], oscillator_spaces_y[0,:i])))
    oscillator_spaces[0].set_offsets(plotdata)


    # between motor oscillators
    oscillator_spaces_x[1,i] = (phases[2] - phases[3]) % 2 * np.pi # absolute phase modulo 2pi
    oscillator_spaces_y[1,i] = ((phases[2] - phases[3]) - (previous_phases[2] - previous_phases[3])) * fs # phase change
    #oscillator_spaces[1].set_xdata(oscillator_spaces_x[1,:])
    #oscillator_spaces[1].set_ydata(oscillator_spaces_y[1,:])
    
   # ax3.scatter(oscillator_spaces_x[0,:], oscillator_spaces_y[1,i], alpha=0.3)
    plotdata = np.transpose(np.asarray((oscillator_spaces_x[1,:i], oscillator_spaces_y[1,:i])))
    oscillator_spaces[1].set_offsets(plotdata)


    # plot the agent position in the environment
    agent_position_x.append(agent_position[0])
    agent_position_y.append(agent_position[1])
    line.set_xdata(agent_position_x)
    line.set_ydata(agent_position_y)

    # plot sensor and oscillator values
    sensory_phase_difference.append(oscillator_spaces_x[0,i])
    sensor_left_input.append(agent.stimulus_sensitivity * agent.stimulus_change_left)
    sensor_right_input.append(agent.stimulus_sensitivity * agent.stimulus_change_right)
    motor_phase_difference.append(oscillator_spaces_x[1,i])

    xdata = range(len(sensory_phase_difference))
    sensory_phase_line.set_ydata(sensory_phase_difference)
    sensory_phase_line.set_xdata(xdata)
    sensor_left_line.set_ydata(sensor_left_input)
    sensor_left_line.set_xdata(xdata)
    sensor_right_line.set_ydata(sensor_right_input)
    sensor_right_line.set_xdata(xdata)
    motor_phase_line.set_ydata(motor_phase_difference)
    motor_phase_line.set_xdata(xdata)


    return line, line_1_2, line_1_3, line_1_4, line_2_3, line_2_4, line_3_4, oscillator_space_1, oscillator_space_2, sensory_phase_line, sensor_left_line, sensor_right_line, motor_phase_line

anim = animation.FuncAnimation(fig, update_simulation, frames = range(duration * fs), interval = 20,
        blit = True)
plt.show()
#anim.save('linechart.gif')


