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
from utils import symmetric_matrix, eucl_distance, eucl_distance_np
from environment import Environment, Social_environment, Social_stimulus_environment
from agent_RL import Gina, Guido, SocialGuido
from visualizations import multi_agent_animation

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
import tkinter as tk
import random
import pickle
import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate_parameters(env, n_episodes, sensitivity, k, f_sens, f_motor, a_sens, a_ips, a_con, a_motor, a_soc_senc, a_soc_motor, agent_coupling, n_agents, flavour, stimulus_ratio, plot, save_data):
    """
    creates n_episode runs for a number of agents with the same combination

    """
    if flavour == 1:
        frequency = np.array([f_sens, f_motor, f_sens])
        phase_coupling = np.array([a_sens, a_con, a_ips, a_motor, a_soc_senc, a_soc_motor])
    else:
        phase_coupling = np.array([a_sens, a_con, a_ips, a_motor])
        frequency = np.array([f_sens, f_motor])


    average_final_distance = np.zeros((3, n_episodes)) # to the two stimuli

    # create multiple agents with same parameters
    agents = []
    for i in range(n_agents):
        agent_id = i
        if flavour == 1:
            policy = SocialGuido(device, fs, frequency, phase_coupling, k, agent_coupling, n_agents, i).to(device)
        else:
            policy = Guido(device, fs, frequency, phase_coupling, k).to(device)
        agents.append(policy)

    approach_scores = []

    for i in range(n_episodes):

        # reset the environment 
        starting_positions = [] 
        starting_orientations = []
        orientations = np.random.uniform(- 0.25 * np.pi, 0.25 * np.pi, n_agents)
        for a in range(n_agents):
            starting_positions.append(np.array([0, np.random.uniform(-105, -95)]))
            starting_orientations.append(orientations[a])
        states = env.reset(starting_positions, starting_orientations, n_agents)
        start_distance = np.mean(env.distances)

        # reset the agent phases
        for a in range(n_agents):
            if flavour == 1:
                agents[a].reset(torch.tensor([0., 0., 0., 0., 0.]))
            else:
                agents[a].reset(torch.tensor([0., 0., 0., 0.,]))


        # Complete the whole episode
        for t in range(duration * fs):
            actions = []

            # make forward pass through each agent and collect all the agent
            for a in range(len(agents)):
                if flavour == 1:
                    action, log_prob, output_angle = agents[a].act(states[a], np.array(env.agent_orientations), env.inter_agent_distances)
                else:
                    action, log_prob, output_angle = agents[a].act(states[a])
                actions.append(output_angle.cpu().detach().numpy())

            # let all agents perform their next action in the environment
            states, rewards, done = env.step(actions, 10)

            if done:
                break 

        end_distance = np.mean(env.distances)
        approach_score = 1 - (end_distance / start_distance)
        approach_scores.append(approach_score)

        distances_1 = 0
        distances_2 = 0
        inter_distance = 0

        for a in range(n_agents):
            distances_1 +=  eucl_distance_np(np.array([-100, 0]), env.agent_positions[a])
            distances_2 +=  eucl_distance_np(np.array([+100, 0]), env.agent_positions[a])
            for b in range(n_agents):
                 inter_distance += eucl_distance_np(env.agent_positions[b], env.agent_positions[a])

        average_final_distance[0, i] = distances_1 / n_agents
        average_final_distance[1, i] = distances_2 / n_agents
        average_final_distance[2, i] = inter_distance / (n_agents**2)

    if plot == True:

        # make an animation plot
        fig, (ax) = plt.subplots(1,1) # ax2
        fig.set_size_inches(10,10)
        anim = multi_agent_animation(fig, ax, stimulus_ratio, stimulus_decay_rate, stimulus_scale, env.position_x, env.position_y, n_agents, duration, fs)
        #anim.save('StimulusEmittingSimulation.gif')
        plt.show()

    if save_data == True:
        # save the data for these runs
        if flavour == 1:
            with open(r"results/multi_agent_simulation_runs_flavour_1.pickle", "wb") as output_file: 
                pickle.dump(average_final_distance, output_file, protocol=pickle.HIGHEST_PROTOCOL)
        elif flavour == 2:
            with open(r"results/multi_agent_simulation_runs_flavour_2.pickle", "wb") as output_file: 
                pickle.dump(average_final_distance, output_file, protocol=pickle.HIGHEST_PROTOCOL)
        elif flavour == 0:
            with open(r"results/multi_agent_simulation_runs_flavour_0.pickle", "wb") as output_file: 
                pickle.dump(average_final_distance, output_file, protocol=pickle.HIGHEST_PROTOCOL)    





# define variables for environment
fs = 20 # Hertz
duration = 30 # Seconds
stimulus_position = [-100, 0] # m, m
stimulus_decay_rate = 0.02 # in the environment
stimulus_scale = 10 # in the environment
stimulus_sensitivity = 8 # of the agent
starting_position = [0, -100] 
starting_orientation = 0.25*np.pi
movement_speed = 10 #m/s
delta_orientation = 0.2*np.pi # rad/s turning speed
agent_radius = 5
agent_eye_angle = 45
stimulus_ratio = 1.2

# define variables for the agent
k = 5/2
f_sens = 1.
f_motor = 1.
a_sens = 0.02
a_ips = 0.02
a_con = 0.1
a_motor = 0.04
a_soc_sens = 0.1  # flavour 1
a_soc_motor = 1  # flavour 1
agent_coupling = 0.5 # flavour 1
flavour = 2


# define variables for the simulation
n_episodes = 1
n_agents = 10

# create starting positions for agents
starting_positions = [] 
starting_orientations = []

# start with random orientions and positions in a certain range
orientations = np.random.uniform(- 0.25 * np.pi, 0.25 * np.pi, n_agents)
for i in range(n_agents):
    starting_positions.append(np.array([0, np.random.uniform(-105, -95)]))
    starting_orientations.append(orientations[i])

for flavour in range(3):
    if flavour == 1:
        env = Social_environment(fs, duration, stimulus_position, stimulus_decay_rate,
            stimulus_scale, stimulus_sensitivity, starting_positions, starting_orientations, movement_speed, agent_radius, agent_eye_angle, delta_orientation, stimulus_ratio)

    elif flavour == 2:
        agent_stimulus_scale = 0.2
        agent_stimulus_decay_rate = 0.02
        env = Social_stimulus_environment(fs, duration, stimulus_position, stimulus_decay_rate,
            stimulus_scale, stimulus_sensitivity, starting_positions, starting_orientations, movement_speed, agent_radius, agent_eye_angle, delta_orientation, agent_stimulus_scale, agent_stimulus_decay_rate, stimulus_ratio)

    elif flavour == 0:
        agent_stimulus_scale = 0
        agent_stimulus_decay_rate = 0
        env = Social_stimulus_environment(fs, duration, stimulus_position, stimulus_decay_rate,
            stimulus_scale, stimulus_sensitivity, starting_positions, starting_orientations, movement_speed, agent_radius, agent_eye_angle, delta_orientation, agent_stimulus_scale, agent_stimulus_decay_rate, stimulus_ratio)

evaluate_parameters(env, n_episodes, stimulus_sensitivity, k, f_sens, f_motor, a_sens, a_ips, a_con, a_motor, a_soc_sens, a_soc_motor, agent_coupling, n_agents, flavour, stimulus_ratio, True, False)

