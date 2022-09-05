#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : multi_agent_simulation.py
# description     : Let several pytorch agents run in the same environment
# author          : Nicolas Coucke
# date            : 2022-09-5
# version         : 1
# usage           : multi_agent_simulation.py
# notes           : install the packages with "pip install -r requirements.txt"
# python_version  : 3.9.2
# ==============================================================================



from utils import symmetric_matrix, eucl_distance
from environment import Environment, Social_environment
from agent_RL import Gina, Guido

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
import tkinter as tk
import random
import pickle

import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")









# define variables for environment
fs = 30 # Hertz
duration = 100 # Seconds
stimulus_position = [0, 0] # m, m
stimulus_decay_rate = 0.02 # in the environment
stimulus_scale = 10 # in the environment
stimulus_sensitivity = 2 # of the agent
starting_position = [0, -100] 
starting_orientation = 0 
movement_speed = 10 #m/s
delta_orientation = 0.5*np.pi # rad/s turning speed
agent_radius = 5
agent_eye_angle = 45





def evaluate_parameters(env, n_episodes, sensitivity, k, f_sens, f_motor, a_sens, a_ips, a_con, a_motor, n_agents):

    frequency = np.array([f_sens, f_motor])
    phase_coupling = np.array([a_sens, a_con, a_ips, a_motor])

    # create multiple agents with same parameters
    agents = []
    for i in range(n_agents):
        policy = Guido(device, fs, frequency, phase_coupling, k).to(device)
        agents.append(policy)
    approach_scores = []
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)

    for i in range(n_episodes):
        # reset the environment 
        starting_positions = [] 
        starting_orientations = []

        # get new starting positions
        for a in range(n_agents):
            starting_positions.append(np.array([0, -100+10*i]))
            starting_orientations.append(0)

        states = env.reset(starting_positions, starting_orientations, n_agents)


        # reset the agent phases
        for i in range(n_agents):
            agents[i].reset(torch.tensor([0., 0., 0, np.pi]))

        # Complete the whole episode
        start_distance = np.mean(env.distances)
        input_values = np.zeros((2, fs * duration))
        phase_differences = np.zeros((4, fs * duration))
        angles = []

        for t in range(duration * fs):
            # do a forward pass for all the angents
            # print("t " + str(t))
            actions = []
            for a in range(len(agents)):
                action, log_prob, output_angle = agents[a].act(states[a])

                actions.append(output_angle.cpu().detach().numpy())
            
            # let all agents perform their next action in the environment
            states, rewards, done = env.step(actions, 10)
            if done:
                break 

            input_values[:, t] = agents[a].input.cpu().detach().numpy()
            phase_differences[:, t] = agents[a].phase_difference.cpu().detach().numpy() * env.fs
            actions.append(env.agent_orientations[a])
            angles.append(policy.output_angle.cpu().detach().numpy())
        
        end_distance = np.mean(env.distances)
        approach_score = 1 - (end_distance / start_distance)
        approach_scores.append(approach_score)


    # plot the trajectory of all agents in the environment
    for i in range(n_agents):
        agents[i].reset(torch.tensor([0., 0., 0, 0.]))
        x_prev = env.position_x[i][0]
        y_prev = env.position_y[i][0]
        t = 0
        for x, y in zip(env.position_x[i], env.position_y[i]):
            # later samples are more visible
            a = 0.1 + 0.5*t/len(env.position_x[i])
            ax1.plot([x_prev, x], [y_prev, y], alpha = a, color = 'red')
            x_prev = x
            y_prev = y
            t+=1
        ax1.set_xlim([-150, 150])
        ax1.set_ylim([-150, 150])



    # for now, plot the inner states of only one agent
    ax3.plot(actions[2:len(actions)], color = 'orange')

    ax3.plot(phase_differences[0, 2:len(angles)], color = 'blue')
    ax3.plot(phase_differences[1, 2:len(angles)], color = 'lightblue')

    ax3.plot(phase_differences[2, 2:len(angles)], color = 'green')
    ax3.plot(phase_differences[3, 2:len(angles)], color = 'lightgreen')

    ax3.plot(input_values[0, 2:len(angles)], color = 'blue', linewidth = 0.8, linestyle = '--')
    ax3.plot(input_values[1, 2:len(angles)], color = 'lightblue', linewidth = 0.8, linestyle = '--')


    # plot the output angle and actions
    ax4.plot(angles)
    ax4.plot(actions)

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
    ax1.set_title('Approach score = ' + str(np.mean(approach_scores)) )
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


sensitivity = 2
k = 2
f_sens = 1.
f_motor = 1.
a_sens = 0.
a_ips = 0.
a_con = 0.5
a_motor = 0.25
n_episodes = 1

n_agents = 1

# create starting positions for agents
starting_positions = [] 
starting_orientations = []

for i in range(n_agents):
    starting_positions.append(np.array([0, -100]))
    starting_orientations.append(0)

env = Social_environment(fs, duration, stimulus_position, stimulus_decay_rate,
     stimulus_scale, stimulus_sensitivity, starting_positions, starting_orientations, movement_speed, agent_radius, agent_eye_angle, delta_orientation)

evaluate_parameters(env, n_episodes, sensitivity, k, f_sens, f_motor, a_sens, a_ips, a_con, a_motor, n_agents)
