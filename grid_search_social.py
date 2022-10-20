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



from utils import symmetric_matrix, eucl_distance, eucl_distance_np
from environment import Environment, Social_environment
from visualizations import single_agent_animation, plot_single_agent_run
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


def perform_grid_search(n_episodes, sensitivity_range, fixed_parameters, asymmetry_range, n_oscillators, n_agents, agent_flavour, social_sensitivity_range, environment, stimulus_ratio, delta_orientation):

   
    n_configurations = len(sensitivity_range)*len(k_range)*len(f_sens_range)*len(f_motor_range)*n_connectivity_variations*len(connection_scaling_factor)*len(asymmetry_range)

    # initialize dataframe to store results
    config = 1 
    if n_oscillators == 4:
        grid_results = pd.DataFrame(columns = ["sensitivity", "social_sensitivity", "k", "f_sens", "f_motor", "a_sens", "a_ips_left", "a_ips_right", "a_con_left", "a_con_right", "a_motor", "scaling_factor", "asymmetry_degree", "start_distance", "start_orientation", "performance"])
    else:
        grid_results = pd.DataFrame(columns = ["sensitivity", "social_sensitivity", "k", "f_sens", "f_motor", "f_soc", "a_sens", "a_ips_left", "a_ips_right", "a_con_left", "a_con_right", "a_motor", "a_soc_sens_left", "a_soc_sens_right", "a_soc_motor_left", "a_soc_motor_right", "scaling_factor", "asymmetry_degree", "start_distance", "start_orientation", "performance"])

    starting_orientations = np.linspace(0, 2*np.pi, n_episodes)

    starting_distances = [20, 40, 60, 80, 100]
    starting_orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]

    if environment == "single_stimulus":
        stimulus_positions = [np.array([0, 0])]
        stimulus_ratio = 0
    elif environment == "double_stimulus":
        stimulus_positions = [np.array([-100, 0]), np.array([100, 0])]
    else:
        print('choose valid environment')

    tq = tqdm(desc='grid_search_' + environment + '_' + str(n_oscillators), total = n_configurations)

    # take the parameters from the structure we got of the other grid search

    print(fixed_parameters)

    # loop through all parameter combinations
    for sensitivity in sensitivity_range:
            # use the of certain sensitivity for all runs
        env = Social_environment(fs, duration, stimulus_position, stimulus_decay_rate,
              stimulus_scale, stimulus_sensitivity, movement_speed, agent_radius, agent_eye_angle, delta_orientation, stimulus_ratio, n_agents)
        for social_sensitivity in social_sensitivity_range:
            for asymmetry_degree in asymmetry_range:

                    scale = fixed_parameters['scaling_factor']
                    k = fixed_parameters['k']
                    f_sens = fixed_parameters['f_sens']
                    f_motor = fixed_parameters['f_motor']
                    a_ips_left = fixed_parameters['a_ips_left']
                    a_ips_right = complementary_connection(a_ips_left, asymmetry_degree)
                    a_con_left = fixed_parameters['a_ips_right']
                    a_con_right = complementary_connection(a_con_left, asymmetry_degree)
                    a_sens = fixed_parameters['a_sens']
                    a_motor = fixed_parameters['a_motor']

                    if n_oscillators == 5:
                        # also determine connections to the 5th oscillator
                        a_soc_sens_left = fixed_parameters['a_soc_sens_left']
                        a_soc_sens_right = complementary_connection(a_soc_sens_left, asymmetry_degree)

                        a_soc_motor_left =  fixed_parameters['a_soc_motor_left']
                        a_soc_motor_right = complementary_connection(a_soc_motor_left, asymmetry_degree)
                        f_soc = f_motor

                        intrinsic_frequencies = np.array([f_sens, f_motor, f_soc])
                        coupling_weights = scale * np.array([ a_sens, a_ips_left, a_ips_right, a_con_left, a_con_right, a_motor,
                                    a_soc_sens_left, a_soc_sens_right, a_soc_motor_left, a_soc_motor_right])
                    else:
                        intrinsic_frequencies = np.array([f_sens, f_motor])
                        coupling_weights = np.array([ a_sens*scale, a_ips_left*scale, a_ips_right*scale, a_con_left*scale, a_con_right*scale, a_motor*scale])

                    # test performance of all parameters
                    performances = evaluate_parameters(env, starting_distances, starting_orientations, k, intrinsic_frequencies, coupling_weights, social_sensitivity, n_oscillators, agent_flavour, 10, False)
                    # save the average score for this configuration
                    i = 0
                    for distance in starting_distances:
                        for orientation in starting_orientations:
                            performance = performances[i]
                            if n_oscillators == 4:
                                configuration_result = pd.Series(data=[sensitivity, social_sensitivity, k, f_sens, f_motor, a_sens, a_ips_left, a_ips_right, a_con_left, a_con_right, a_motor, scale, asymmetry_degree, distance, orientation, performance],  index= grid_results.columns, name = config)
                            else:
                                configuration_result = pd.Series(data=[sensitivity, social_sensitivity, k, f_sens, f_motor, f_soc, a_sens, a_ips_left, a_ips_right, a_con_left, a_con_right, a_motor, a_soc_sens_left, a_soc_sens_right, a_soc_motor_left, a_soc_motor_right, scale, asymmetry_degree, distance, orientation, performance],  index= grid_results.columns, name = config)
                            i+=1
                            grid_results = grid_results.append(configuration_result)
                    tq.update(1)
                    #print( "configuration " + str(config) + " of " + str(n_configurations))
                    config+=1

            # save the results every few runs
            with open(r"GridSearchResults_social.pickle", "wb") as output_file: 
                pickle.dump(grid_results, output_file, protocol=pickle.HIGHEST_PROTOCOL)
    print("done")                          
    return grid_results
         




# define variables for environment
fs = 50 # Hertz
duration = 20 # Seconds
stimulus_position = [0, 0] # m, m
stimulus_decay_rate = 0.02 # in the environment
stimulus_scale = 1 # in the environment
stimulus_sensitivity = 5 # of the agent
starting_position = [0, -100] 
starting_orientation = -0.25*np.pi
movement_speed = 10 #m/s
delta_orientation = 0.2*np.pi # rad/s turning speed
agent_radius = 5
agent_eye_angle = 45

# define the parameters for the grid search
sensitivity_range = [1.]#[1., 5., 10.] #np.arange(1, 20, 1)
social_sensitivity_range = [1, 2, 3, 4, 5]
k_range = [5., 4., 2.]
f_sens_range = [0, 1.]#[1., 1.3, 1.6] #np.arange(0.3, 3, 0.3)
f_motor_range = [0., 1.]
connection_scaling_factor = np.arange(0.5, 2, 0.5)
n_connectivity_variations = 50
asymmetry_range = np.arange(0, 1, 0.25)
n_episodes = 25
n_oscillators = 4
environment = "single_stimulus"
stimulus_ratio = 1
n_agents = 10

# extract variables from bash script
if __name__ == "__main__":
   parser = argparse.ArgumentParser(
        description='grid_search_social', formatter_class=argparse.RawDescriptionHelpFormatter)
   # prepare parameters
   parser.add_argument("--environment", type=str,
                        default='single_stimulus')
   parser.add_argument("--n_oscillators", type=int,
                     default=4)
   parser.add_argument("--agent_flavour", type=str,
                     default="eco")
   all_args = parser.parse_known_args(sys.argv[1:])[0]
  
   environment = all_args.environment
   n_oscillators = all_args.n_oscillators
   agent_flavour = all_args.agent_flavour

print(environment)
print(n_oscillators)

fixed_parameters = {"sensitivity": 5, "k": 5., "f_sens": 1., "f_motor": 1., "f_soc": 1., "a_sens": 0.5, "a_ips_left": 0.5, "a_ips_right": 0.5, "a_con_left": 0.4, "a_con_right": 0.5, "a_soc_sens_left": 0.5, "a_soc_sens_right": 0.5, "a_soc_motor_left": 0.5, "a_soc_motor_right": 0.5,  "a_motor": 0.5, "scaling_factor": 0.1,}


# execute the grid search
perform_grid_search(n_episodes, sensitivity_range, fixed_parameters, asymmetry_range, n_oscillators, n_agents, agent_flavour, social_sensitivity_range, environment, stimulus_ratio, delta_orientation)

