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



from utils import symmetric_matrix, eucl_distance, initiate_coupling_weights
from environment import Environment, Social_environment
from simulations import evaluate_parameters

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


def perform_grid_search(n_episodes, sensitivity_range, k_range, f_sens_range, f_motor_range, n_connectivity_variations, connection_scaling_factor, asymmetry_range, n_oscillators, environment, stimulus_ratio):
   
   
   n_configurations = len(sensitivity_range)*len(k_range)*len(f_sens_range)*len(f_motor_range)*n_connectivity_variations*len(connection_scaling_factor)*len(asymmetry_range)

   # initialize dataframe to store results
   config = 1 
   if n_oscillators == 4:
      grid_results = pd.DataFrame(columns = ["sensitivity", "k", "f_sens", "f_motor", "a_sens", "a_ips_left", "a_ips_right", "a_con_left", "a_con_right", "a_motor", "scaling_factor", "asymmetry_degree", "start_distance", "start_orientation", "performance"])
   else:
      grid_results = pd.DataFrame(columns = ["sensitivity", "k", "f_sens", "f_motor", "f_soc", "a_sens", "a_ips_left", "a_ips_right", "a_con_left", "a_con_right", "a_motor", "a_soc_sens_left", "a_soc_sens_right", "a_soc_motor_left", "a_soc_motor_right", "scaling_factor", "asymmetry_degree", "start_distance", "start_orientation", "performance"])

   starting_orientations = np.linspace(0, 2*np.pi, n_episodes)

   starting_distances = [100]
   starting_orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]

   if environment == "single_stimulus":
      stimulus_positions = [np.array([0, 0])]
      stimulus_ratio = 0
   elif environment == "double_stimulus":
      stimulus_positions = [np.array([-100, 0]), np.array([100, 0])]
   else:
      print('choose valid environment')

   tq = tqdm(desc='grid_search_' + environment + '_' + str(n_oscillators), total = n_configurations)
   # loop through all parameter combinations
   for sensitivity in sensitivity_range:
      # use the of certain sensitivity for all runs
      env = Environment(fs, duration, stimulus_positions, stimulus_ratio, stimulus_decay_rate,
         stimulus_scale, sensitivity, starting_position, starting_orientation, movement_speed, agent_radius, agent_eye_angle, delta_orientation)
      #env = Social_environment(fs, duration, stimulus_position, stimulus_decay_rate,
        #stimulus_scale, stimulus_sensitivity, movement_speed, agent_radius, agent_eye_angle, delta_orientation, stimulus_ratio)
      for k in k_range:
         for f_sens in f_sens_range:
            for f_motor in f_motor_range:
               for variation in range(n_connectivity_variations):
                  for scale in connection_scaling_factor:
                     for asymmetry_degree in asymmetry_range:
                        
                        if n_oscillators == 4:
                           intrinsic_frequencies = np.array([f_sens, f_motor])
                           coupling_weights, a_sens, a_ips_left, a_ips_right, a_con_left, a_con_right, a_motor = initiate_coupling_weights(scale, asymmetry_degree, True, n_oscillators)

                        else:
                           intrinsic_frequencies = np.array([f_sens, f_motor, f_motor])
                           coupling_weights, a_sens, a_ips_left, a_ips_right, a_con_left, a_con_right, a_motor, a_soc_sens_left, a_soc_sens_right, a_soc_motor_left, a_soc_motor_right = initiate_coupling_weights(scale, asymmetry_degree, True, n_oscillators)

                        # test performance of all parameters
                        performances = evaluate_parameters(env, starting_distances, starting_orientations, k, intrinsic_frequencies, coupling_weights, n_oscillators, False)
                        # save the average score for this configuration
                        i = 0
                        for distance in starting_distances:
                           for orientation in starting_orientations:
                              performance = performances[i]
                              if n_oscillators == 4:
                                 configuration_result = pd.Series(data=[sensitivity, k, f_sens, f_motor, a_sens, a_ips_left, a_ips_right, a_con_left, a_con_right, a_motor, scale, asymmetry_degree, distance, orientation, performance],  index= grid_results.columns, name = config)
                              else:
                                 configuration_result = pd.Series(data=[sensitivity, k, f_sens, f_motor, f_motor, a_sens, a_ips_left, a_ips_right, a_con_left, a_con_right, a_motor, a_soc_sens_left, a_soc_sens_right, a_soc_motor_left, a_soc_motor_right, scale, asymmetry_degree, distance, orientation, performance],  index= grid_results.columns, name = config)
                              i+=1
                              grid_results = grid_results.append(configuration_result)
                        tq.update(1)
                        #print( "configuration " + str(config) + " of " + str(n_configurations))
                        config+=1

         # save the results every few runs
         with open(r"GridSearchResults_asymmetric.pickle", "wb") as output_file: 
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

# extract variables from bash script
if __name__ == "__main__":
   parser = argparse.ArgumentParser(
        description='grid_search', formatter_class=argparse.RawDescriptionHelpFormatter)
   # prepare parameters
   parser.add_argument("--environment", type=str,
                        default='single_stimulus')
   parser.add_argument("--n_oscillators", type=int,
                     default=4)
   all_args = parser.parse_known_args(sys.argv[1:])[0]
  
   environment = all_args.environment
   n_oscillators = all_args.n_oscillators


# execute the grid search
perform_grid_search(n_episodes, sensitivity_range, k_range, f_sens_range, f_motor_range, n_connectivity_variations, connection_scaling_factor, asymmetry_range,  n_oscillators, environment, stimulus_ratio)
