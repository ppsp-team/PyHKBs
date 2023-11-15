#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : grid_search_social.py
# description     : use the Pytorch agent and to explore the different parameter values specific to the social agents
# author          : Nicolas Coucke
# date            : 2022-10-14
# version         : 1
# usage           : grid_search_social.py
# notes           : install the packages with "pip install -r requirements.txt"
# python_version  : 3.9.2
# ==============================================================================



from utils import symmetric_matrix, eucl_distance, eucl_distance_np, initiate_coupling_weights
from environment import Social_environment, Social_stimulus_environment
from visualizations import single_agent_animation, plot_single_agent_run
from simulations import evaluate_parameters_social
from agent_RL import Gina, Guido, MultipleGuidos, SocialGuido

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
import tkinter as tk
import random
import pickle
import sys
import argparse
from tqdm import tqdm


import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(np.arange(0, 1, 0.5))
def complementary_connection(connection_strength, asymmetry_degree):
   return (1 - connection_strength) * asymmetry_degree + (1 - asymmetry_degree) * connection_strength

def perform_grid_search(n_episodes, sensitivity_scale, k_range, f_sens_range, f_motor_range, a_motor_range, internal_connection_scale, asymmetry_range, n_oscillators, environment, stimulus_ratio_range, starting_distances, starting_orientations, random_phases, n_agents, agent_flavour, social_sensitivity_scale, social_weight_decay_rate):
   
    triangle_samples = 51 # 0 to 50 including 0
    # the number of combinations is equal to the 'triangle number'
    n_triangle_configurations = int(triangle_samples*(triangle_samples + 1) / 2)

    n_configurations = n_triangle_configurations*len(k_range)*len(f_sens_range)*len(f_motor_range)*len(a_motor_range)*len(asymmetry_range)*len(stimulus_ratio_range)
    print(n_configurations)

    # initialize dataframe to store results
    config = 1 
   
    config = 1 
    if n_oscillators == 4:
        grid_results = pd.DataFrame(columns = ["stimulus_sensitivity", "social_sensitivity", "internal_connectivity", "k", "f_sens", "f_motor", "a_sens", "a_ips_left", "a_ips_right", "a_con_left", "a_con_right", "a_motor", "asymmetry_degree", "start_distance", "start_orientation", "performance"])
    else:
        grid_results = pd.DataFrame(columns = ["stimulus_sensitivity", "social_sensitivity", "internal_connectivity",  "k", "f_sens", "f_motor", "f_soc", "a_sens", "a_ips_left", "a_ips_right", "a_con_left", "a_con_right", "a_motor", "a_soc_sens_left", "a_soc_sens_right", "a_soc_motor_left", "a_soc_motor_right", "asymmetry_degree", "start_distance", "start_orientation", "performance"])

    #starting_orientations = np.linspace(0, 2*np.pi, n_episodes)
    starting_orientation = 0

    if environment == "single_stimulus":
        stimulus_positions = [np.array([0, 0])]
        stimulus_ratio = 0
    elif environment == "double_stimulus":
        stimulus_positions = [np.array([-100, 0]), np.array([100, 0])]
    else:
        print('choose valid environment')

    tq = tqdm(desc='grid_search_' + environment + '_' + str(n_oscillators), total = n_configurations)

    # make a list that stores one run dictionary per entry in the pandas dataframe
    grid_runs = []
    i_run = 0
    # loop through all parameter combinations

    for stimulus_ratio in stimulus_ratio_range:
        for stimulus_sensitivity in range(triangle_samples):
            for social_sensitivity in range(stimulus_sensitivity, triangle_samples):
                internal_coupling = triangle_samples - 1 - social_sensitivity # if social sensitivity is 50 then internal coupling should be 0
                
                scale = internal_coupling * internal_connection_scale
                sensitivity = stimulus_sensitivity * sensitivity_scale
                social = social_sensitivity * social_sensitivity_scale
                            # use the of certain sensitivity for all runs
                if agent_flavour == 'social':
                    env = Social_environment(fs, duration, stimulus_positions, stimulus_decay_rate,
                        stimulus_scale, stimulus_sensitivity, movement_speed, agent_radius, agent_eye_angle, delta_orientation, stimulus_ratio, n_agents)
                elif agent_flavour == 'eco':
                    agent_stimulus_scale = 0.01 * social_sensitivity_scale
                    agent_stimulus_decay_rate = social_weight_decay_rate
                    env = Social_stimulus_environment(fs, duration, stimulus_positions, stimulus_decay_rate,
                        stimulus_scale, sensitivity, movement_speed, agent_radius, agent_eye_angle, delta_orientation, agent_stimulus_scale, agent_stimulus_decay_rate, stimulus_ratio, n_agents)

                for k in k_range:
                    for f_sens in f_sens_range:
                        for f_motor in f_motor_range:
                            for a_motor in a_motor_range:
                                for asymmetry_degree in asymmetry_range:
                                    # define the degree of influence for internal coupling
                                    
                                    if n_oscillators == 4:
                                        intrinsic_frequencies = np.array([f_sens, f_motor])
                                        coupling_weights, a_sens, a_ips_left, a_ips_right, a_con_left, a_con_right, a_motor = initiate_coupling_weights(scale, asymmetry_degree, False, n_oscillators, a_sens = 0, a_motor = a_motor, a_ips = 0, a_con = 1)
                                    else:
                                        intrinsic_frequencies = np.array([f_sens, f_motor, f_motor])
                                        coupling_weights, a_sens, a_ips_left, a_ips_right, a_con_left, a_con_right, a_motor, a_soc_sens_left, a_soc_sens_right, a_soc_motor_left, a_soc_motor_right = initiate_coupling_weights(scale, asymmetry_degree, False, n_oscillators, a_sens = 0, a_motor = a_motor, a_ips = 0, a_con = 1,  a_soc_sens = 0, a_soc_motor = 1)


                                    # test performance of all parameters
                                    runs = evaluate_parameters_social(env, device, fs, duration, starting_distances, starting_orientations, k, intrinsic_frequencies, coupling_weights, social, social_weight_decay_rate, n_oscillators, agent_flavour, n_agents, False)
                                    # save the average score for this configuration
                                    i = 0
                                    for distance in starting_distances:
                                        for orientation in starting_orientations:
                                            run = runs[i]
                                            performance = run['approach score']
                                            if n_oscillators == 4:
                                                configuration_result = pd.Series(data=[sensitivity, social, scale, k, f_sens, f_motor, a_sens, a_ips_left, a_ips_right, a_con_left, a_con_right, a_motor, asymmetry_degree, distance, orientation, performance],  index= grid_results.columns, name = config)
                                            else:
                                                configuration_result = pd.Series(data=[sensitivity, social, scale, k, f_sens, f_motor, f_motor, a_sens, a_ips_left, a_ips_right, a_con_left, a_con_right, a_motor, a_soc_sens_left, a_soc_sens_right, a_soc_motor_left, a_soc_motor_right, asymmetry_degree, distance, orientation, performance],  index= grid_results.columns, name = config)
                                            i+=1
                                            # store one entry in the dataframe and one in the list
                                            grid_results = grid_results.append(configuration_result)
                                            with open(r"results/Social_GridSearchResults_" + str(n_oscillators) + '_' + agent_flavour + "/run_" + str(i_run) + ".pickle", "wb") as output_file: 
                                                pickle.dump(run, output_file, protocol=pickle.HIGHEST_PROTOCOL)
                                            i_run+=1
                                    tq.update(1)
                                    #print( "configuration " + str(config) + " of " + str(n_configurations))
                                    config+=1

                                    # save the results every few runs
                                    with open(r"results/Social_GridSearchResults_" + str(n_oscillators) + '_' + agent_flavour + ".pickle", "wb") as output_file: 
                                        pickle.dump(grid_results, output_file, protocol=pickle.HIGHEST_PROTOCOL)
    print("done")                          
    return grid_results
            


fs = 100# Hertz
duration = 30 # Seconds
stimulus_positions = [np.array([-100, 0]), np.array([100,0])] # m, m
stimulus_decay_rate = 0.02 # in the environment
stimulus_scale = 1 # in the environment
stimulus_sensitivity = 1 # of the agent
movement_speed = 10 #m/s
delta_orientation = 0.1*np.pi # rad/s turning speed # not used anymore here
stimulus_ratio_range = [0.8]
agent_radius = 2.5
agent_eye_angle = 0.5 * np.pi # 45 degrees
starting_position = [0, -100]
starting_distances = [100] 
starting_orientations = [0.2]
environment = "double_stimulus"
random_phases = False

# define the parameters for the grid search
k_range = [2.]
f_sens_range = [5.]#[1., 1.3, 1.6] #np.arange(0.3, 3, 0.3)
f_motor_range = [5.]
a_motor_range = [1.]
asymmetry_range = [0.]#np.arange(0, 1., 0.1)


sensitivity_scale = 10
social_sensitivity_scale = 5
internal_connection_scale = 5
social_weight_decay_rate = 0.01
n_episodes = 1
n_agents = 10

n_oscillators = 4
agent_flavour = 'eco'
perform_grid_search(n_episodes, sensitivity_scale, k_range, f_sens_range, f_motor_range, a_motor_range, internal_connection_scale, asymmetry_range, n_oscillators, environment, stimulus_ratio_range, starting_distances, starting_orientations, random_phases, n_agents, agent_flavour, social_sensitivity_scale, social_weight_decay_rate)

n_oscillators = 5
agent_flavour = 'eco'
perform_grid_search(n_episodes, sensitivity_scale, k_range, f_sens_range, f_motor_range, a_motor_range, internal_connection_scale, asymmetry_range, n_oscillators, environment, stimulus_ratio_range, starting_distances, starting_orientations, random_phases, n_agents, agent_flavour, social_sensitivity_scale, social_weight_decay_rate)
agent_flavour = 'social'
perform_grid_search(n_episodes, sensitivity_scale, k_range, f_sens_range, f_motor_range, a_motor_range, internal_connection_scale, asymmetry_range, n_oscillators, environment, stimulus_ratio_range, starting_distances, starting_orientations, random_phases, n_agents, agent_flavour, social_sensitivity_scale, social_weight_decay_rate)

# execute the grid search

