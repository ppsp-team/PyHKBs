#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : single_agent_simulation.py
# description     : Simulate an agent that moves towards a stimulus in the environment
# author          : Nicolas Coucke
# date            : 2022-07-14
# version         : 1
# usage           : python single_agent_simulation.py
# notes           : install the packages with "pip install -r requirements.txt"
# python_version  : 3.9.2
# ==============================================================================


import matplotlib.pyplot as plt
import torch
import numpy as np
from utils import symmetric_matrix, eucl_distance
from agent import Agent
import time


# define variables of simulated environment
fs = torch.as_tensor([25])
duration = torch.as_tensor([50]) # s
stimulus_position = torch.tensor([-75., 75.]) # [m, m]
stimulus_gradient = torch.as_tensor([3])
max_stimulus_value = torch.as_tensor([100.])
periodic_randomization = True

# instantiate an agent
agent_id = 1
stimulus_sensitivity = torch.as_tensor([10.])
phase_coupling_matrix = symmetric_matrix(5, 4)
anti_phase_coupling_matrix = symmetric_matrix(5, 4)
initial_phases = torch.tensor([0., torch.pi, 0.6, torch.pi]) # rad
frequencies = torch.tensor([1.2, 1.3, 1.5, 1.5]) # Hertz
movement_speed = torch.as_tensor([30]) # m/s
agent_position = torch.tensor([0., 0.]) # [m, m]
agent_orientation = torch.as_tensor([0.]) # rad

# create agent class
agent = Agent(agent_id, stimulus_sensitivity, phase_coupling_matrix, anti_phase_coupling_matrix,
            initial_phases, frequencies, movement_speed, agent_position, agent_orientation)

# generate time and initialize arrays to store trajectory
t = torch.linspace(start=0, end=float(duration), steps=int(duration * fs))
agent_position_x = np.zeros((len(t),))
agent_position_y = np.zeros((len(t),))

# perform simulation
for i in range(len(t)):
    # get the current eye positions of the agent
    left_eye_position, right_eye_position = agent.eye_positions()

    # calculate the stimulus intensity at the eye positions
    stimulus_intensity_left = max_stimulus_value - stimulus_gradient * eucl_distance(stimulus_position, left_eye_position)
    stimulus_intensity_right = max_stimulus_value - stimulus_gradient * eucl_distance(stimulus_position, right_eye_position)

    # get agent movement based on stimulus intensities
    agent_position, agent_orientation,_ ,_ = agent.next_timestep(i, stimulus_intensity_left, stimulus_intensity_right, periodic_randomization)

    # save agent position for visualization
    agent_position_x[i] = agent_position[0]
    agent_position_y[i] = agent_position[1]

# visualize agent trajectory
plt.plot(agent_position_x, agent_position_y)
plt.scatter(stimulus_position[0], stimulus_position[1])
plt.xlim([-100,100])
plt.ylim([-100,100])
plt.show()
