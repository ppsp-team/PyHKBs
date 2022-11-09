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


def perform_grid_search(n_episodes, sensitivity_range, k_range, f_sens_range, f_motor_range, a_motor_range, connection_scaling_factor, asymmetry_range, n_oscillators, environment, stimulus_ratio_range, starting_distances, starting_orientations, random_phases):
   
   
   n_configurations = len(sensitivity_range)*len(k_range)*len(f_sens_range)*len(f_motor_range)*len(a_motor_range)*len(connection_scaling_factor)*len(asymmetry_range)*len(stimulus_ratio_range)
   print(n_configurations)

   # initialize dataframe to store results
   config = 1 
   if n_oscillators == 4:
      grid_results = pd.DataFrame(columns = ["sensitivity", "k", "f_sens", "f_motor", "a_sens", "a_ips_left", "a_ips_right", "a_con_left", "a_con_right", "a_motor", "scaling_factor", "asymmetry_degree", "start_distance", "start_orientation", "performance"])
   else:
      grid_results = pd.DataFrame(columns = ["sensitivity", "k", "f_sens", "f_motor", "f_soc", "a_sens", "a_ips_left", "a_ips_right", "a_con_left", "a_con_right", "a_motor", "a_soc_sens_left", "a_soc_sens_right", "a_soc_motor_left", "a_soc_motor_right", "scaling_factor", "asymmetry_degree", "start_distance", "start_orientation", "performance"])

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

   # loop through all parameter combinations
   for sensitivity in sensitivity_range:
      for stimulus_ratio in stimulus_ratio_range:
         # use the of certain sensitivity for all runs
         env = Environment(fs, duration, stimulus_positions, stimulus_ratio, stimulus_decay_rate,
            stimulus_scale, sensitivity, starting_position, starting_orientation, movement_speed, agent_radius, agent_eye_angle, delta_orientation)
         #env = Social_environment(fs, duration, stimulus_position, stimulus_decay_rate,
         #stimulus_scale, stimulus_sensitivity, movement_speed, agent_radius, agent_eye_angle, delta_orientation, stimulus_ratio)
         for k in k_range:
            for f_sens in f_sens_range:
               for f_motor in f_motor_range:
                  for a_motor in a_motor_range:
                     for scale in connection_scaling_factor:
                        for asymmetry_degree in asymmetry_range:
                           
                           if n_oscillators == 4:
                              intrinsic_frequencies = np.array([f_sens, f_motor])
                              coupling_weights, a_sens, a_ips_left, a_ips_right, a_con_left, a_con_right, a_motor = initiate_coupling_weights(scale, asymmetry_degree, False, n_oscillators, a_sens = 0, a_motor = a_motor, a_ips = 0, a_con = 1)
                           else:
                              intrinsic_frequencies = np.array([f_sens, f_motor, f_motor])
                              coupling_weights, a_sens, a_ips_left, a_ips_right, a_con_left, a_con_right, a_motor, a_soc_sens_left, a_soc_sens_right, a_soc_motor_left, a_soc_motor_right = initiate_coupling_weights(scale, asymmetry_degree, False, n_oscillators, a_sens = 0, a_motor = a_motor, a_ips = 0, a_con = 1,  a_soc_sens = 0, a_soc_motor = 1)

                           # test performance of all parameters
                           runs = evaluate_parameters(env, device, duration, fs, starting_distances, starting_orientations, k, intrinsic_frequencies, coupling_weights, n_oscillators, random_phases)
                           # save the average score for this configuration
                           i = 0
                           for distance in starting_distances:
                              for orientation in starting_orientations:
                                 run = runs[i]
                                 performance = run['approach score']
                                 print(end_time)
                                 end_time = run['end time']
                                 if n_oscillators == 4:
                                    configuration_result = pd.Series(data=[sensitivity, k, f_sens, f_motor, a_sens, a_ips_left, a_ips_right, a_con_left, a_con_right, a_motor, scale, asymmetry_degree, distance, orientation, performance, end_time],  index= grid_results.columns, name = config)
                                 else:
                                    configuration_result = pd.Series(data=[sensitivity, k, f_sens, f_motor, f_motor, a_sens, a_ips_left, a_ips_right, a_con_left, a_con_right, a_motor, a_soc_sens_left, a_soc_sens_right, a_soc_motor_left, a_soc_motor_right, scale, asymmetry_degree, distance, orientation, performance, end_time],  index= grid_results.columns, name = config)
                                 i+=1
                                 # store one entry in the dataframe and one in the list
                                 grid_results = grid_results.append(configuration_result)
                                 grid_runs.append(run)
                           tq.update(1)
                           #print( "configuration " + str(config) + " of " + str(n_configurations))
                           config+=1

            search = 'deterministic'
            # save the results every few runs
            if random_phases == True:
               search = 'random'
            with open(r"GridSearchResults_" + str(n_oscillators) + '_' + search + ".pickle", "wb") as output_file: 
               pickle.dump([grid_results, grid_runs], output_file, protocol=pickle.HIGHEST_PROTOCOL)
   print("done")                          
   return grid_results
         

# define variables for environment
fs = 250# Hertz
duration = 30 # Seconds
stimulus_positions = [np.array([-100, 0]), np.array([100,0])] # m, m
stimulus_decay_rate = 0.02 # in the environment
stimulus_scale = 1 # in the environment
stimulus_sensitivity = 1 # of the agent
movement_speed = 10 #m/s
delta_orientation = 0.1*np.pi # rad/s turning speed # not used anymore here
stimulus_ratio_range = [0.]
agent_radius = 2.5
agent_eye_angle = 0.5 * np.pi # 45 degrees
starting_position = [0, -100] 
environment = "double_stimulus"

# define the parameters for the grid search
sensitivity_range = [0., 5.]#[1., 5., 10.] #np.arange(1, 20, 1)
k_range = [2.]
f_sens_range = [5]#[1., 1.3, 1.6] #np.arange(0.3, 3, 0.3)
f_motor_range = [5.]
connection_scaling_factor = np.arange(0.2, 5.2, 0.2)
#n_connectivity_variations = 50
asymmetry_range = [0.]#np.arange(0, 1., 0.1)
print(asymmetry_range)
n_episodes = 100
n_oscillators = 4
environment = "single_stimulus"

stimulus_ratio = 1

a_motor_range = [0., 1.]



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


# first grid search (randomized)
starting_distances = [100]
# we want to make simulations with 100 random intial phases but same angle:
starting_orientations = np.zeros((100,))
random_phases = True
environment = 'single_stimulus'
for n_oscillators in [4, 5]:
   # execute the grid search
   perform_grid_search(n_episodes, sensitivity_range, k_range, f_sens_range, f_motor_range, a_motor_range, connection_scaling_factor, asymmetry_range,  n_oscillators, environment, stimulus_ratio_range, starting_distances, starting_orientations, random_phases)



# second grid search (deterministic)
starting_orientations = [0]
random_phases = False
environment = 'double_stimulus'
sensitivity_range = np.arange(0, 11, 1)
stimulus_ratio_range = [0., 0.8, 1]
asymmetry_range = np.arange(0, 1.1, 0.1)
for n_oscillators in [4, 5]:
   # execute the grid search
   perform_grid_search(n_episodes, sensitivity_range, k_range, f_sens_range, f_motor_range, a_motor_range, connection_scaling_factor, asymmetry_range,  n_oscillators, environment, stimulus_ratio_range, starting_distances, starting_orientations, random_phases)
