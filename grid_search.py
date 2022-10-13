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
from environment import Environment, Social_environment
from visualizations import single_agent_animation, plot_single_agent_run
from agent_RL import Gina, Guido, MultipleGuidos

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


def complementary_connection(connection_strength, asymmetry_degree):
   return (1 - connection_strength) * asymmetry_degree + (1 - asymmetry_degree) * connection_strength


def perform_grid_search(n_episodes, sensitivity_range, k_range, f_sens_range, f_motor_range, n_connectivity_variations, connection_scaling_factor, asymmetry_range, n_oscillators, environment, stimulus_ratio):
   
   
   n_configurations = len(sensitivity_range)*len(k_range)*len(f_sens_range)*len(f_motor_range)*n_connectivity_variations*len(connection_scaling_factor)

   # initialize dataframe to store results
   config = 1 
   if n_oscillators == 4:
      grid_results = pd.DataFrame(columns = ["sensitivity", "k", "f_sens", "f_motor", "a_sens", "a_ips_left", "a_ips_right", "a_con_left", "a_con_right", "a_motor", "scaling_factor", "asymmetry_degree", "start_distance", "start_orientation", "performance"])
   else:
      grid_results = pd.DataFrame(columns = ["sensitivity", "k", "f_sens", "f_motor", "f_soc", "a_sens", "a_ips_left", "a_ips_right", "a_con_left", "a_con_right", "a_motor", "a_soc_sens_left", "a_soc_sens_right", "a_soc_motor_left", "a_soc_motor_right", "scaling_factor", "asymmetry_degree", "start_distance", "start_orientation", "performance"])

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
                        # make random variables for connectivities:
                        a_ips_left = np.random.uniform()
                        a_ips_right = complementary_connection(a_ips_left, asymmetry_degree)

                        a_con_left = np.random.uniform()
                        a_con_right = complementary_connection(a_con_left, asymmetry_degree)

                        a_sens = np.random.uniform()
                        a_motor = np.random.uniform()

                        if n_oscillators == 5:
                           # also determine connections to the 5th oscillator
                           a_soc_sens_left = np.random.uniform()
                           a_soc_sens_right = complementary_connection(a_soc_sens_left, asymmetry_degree)

                           a_soc_motor_left = np.random.uniform()
                           a_soc_motor_right = complementary_connection(a_soc_motor_left, asymmetry_degree)
                           f_soc = f_motor

                           intrinsic_frequencies = np.array([f_sens, f_motor, f_soc])
                           coupling_weights = scale * np.array([ a_sens, a_ips_left, a_ips_right, a_con_left, a_con_right, a_motor,
                                       a_soc_sens_left, a_soc_sens_right, a_soc_motor_left, a_soc_motor_right])
                        else:
                           intrinsic_frequencies = np.array([f_sens, f_motor])
                           coupling_weights = np.array([ a_sens*scale, a_ips_left*scale, a_ips_right*scale, a_con_left*scale, a_con_right*scale, a_motor*scale])

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
                                 configuration_result = pd.Series(data=[sensitivity, k, f_sens, f_motor, f_soc, a_sens, a_ips_left, a_ips_right, a_con_left, a_con_right, a_motor, a_soc_sens_left, a_soc_sens_right, a_soc_motor_left, a_soc_motor_right, scale, asymmetry_degree, distance, orientation, performance],  index= grid_results.columns, name = config)
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
         


def evaluate_grid_search(grid_results):
   """
   look at parameters and find the ones that perform best
   """

   # find all values with a certain performance
   grid_results = grid_results[grid_results["performance"] > 0.5]

   with pd.option_context('display.max_rows', None,
                        'display.precision', 3,
                        ):
      print(grid_results)


def visualize_grid_search(grid_results, x_axis, y_axis, other_parameters):
   
   # extract requested values
   for key in other_parameters:
      if not ( (key == x_axis) or (key == y_axis) ):

         # make subselection of fixed parameters
         grid_results = grid_results[grid_results[key] == other_parameters[key]]

   # for the other parameters, make a numpy array to plot
   x_axis_values = np.sort(np.unique(grid_results[x_axis].to_numpy()))
   y_axis_values = np.sort(np.unique(grid_results[y_axis].to_numpy()))
   plotting_array = np.zeros((len(x_axis_values), len(y_axis_values)))

   
   for x in range(len(x_axis_values)):
      for y in range(len(y_axis_values)):
         plot_val = grid_results[grid_results[x_axis] == x_axis_values[x]]
         print(plot_val)
         plot_val = plot_val[grid_results[y_axis] == y_axis_values[y]]
         print(plot_val)
         plotting_array[x, y] = float(plot_val.performance.to_numpy())


   plt.xticks(np.arange(0, len(x_axis_values), 1), x_axis_values)
   plt.yticks(np.arange(0, len(y_axis_values), 1), y_axis_values)
   plt.xlabel(x_axis)
   plt.ylabel(y_axis)
   plt.imshow(plotting_array)
   plt.colorbar()
   plt.show()



def evaluate_parameters(env, starting_distances, starting_orientations, k, frequency, coupling_weights, n_oscillators, plot):



   # create agent with these parameters
   policy = Guido(device, fs, frequency, coupling_weights, k, n_oscillators).to(device)

   approach_scores = []   

   # do ten episodes for each parameter combination
   for starting_distance in starting_distances:
      for starting_orientation in starting_orientations:
         
         starting_position = np.array([0, -starting_distance])

         state = env.reset(starting_position, starting_orientation)

         # reset Guido
         policy.reset(torch.tensor([0., 0., 0., 0.]))

         # Complete the whole episode
         start_distance = env.distance
         input_values = np.zeros((2, fs * duration))
         phase_differences = np.zeros((4, fs * duration))
         phases = np.zeros((4, fs * duration))

         actions = []
         angles = []
         for t in range(duration * fs):
            #action, log_prob = policy.act(state)
            action, log_prob, output_angle = policy.act(state)
            #state, reward, done = env.step(action, 10)
            state, reward, done = env.step(output_angle.cpu().detach().numpy(), 10)
            if done:
                  break 
            input_values[:, t] = policy.input.cpu().detach().numpy()
            phase_differences[:, t] = policy.phase_difference.cpu().detach().numpy() * env.fs
            phases[:, t] = policy.phases.cpu().detach().numpy()

            actions.append(env.orientation)
            angles.append(policy.output_angle.cpu().detach().numpy())
         end_distance = env.distance
         approach_score = 1 - (end_distance / start_distance)
         approach_scores.append(approach_score)


         if plot == True:
            fig = plot_single_agent_run(f_sens, f_motor, a_sens, a_motor, a_ips, a_con, k, env.position_x, env.position_y, phase_differences, input_values, angles, actions, stimulus_scale, stimulus_decay_rate)
            anim = single_agent_animation(env.position_x, env.position_y, phases, phase_differences, stimulus_scale, stimulus_decay_rate, duration, fs)
            #anim.save('GuidoSimulation.gif')

   return approach_scores


def evaluate_parameters_concatenated(env, n_episodes, k, f_sens, f_motor, a_sens, a_ips, a_con, a_motor, plot):
   """
   evaluate the parameters in a parallelized way
   """
   
   frequency = np.array([f_sens, f_motor])
   phase_coupling = np.array([a_sens, a_con, a_ips, a_motor])

   # create an agent for each episode and concatenate them to parallelize
   policy = MultipleGuidos(device, fs, frequency, phase_coupling, k, n_episodes).to(device)


   # define a posiition/environment for each agent and initiate them in the environment
   starting_positions = [] 
   starting_orientations = []
   orientations = [np.random.uniform(- 0.25 * np.pi, 0.25 * np.pi, n_episodes)]

   start_distances = []
   for a in range(n_episodes):
      start_distance = np.random.uniform(-105, -95)
      start_distances.append(start_distance)
      starting_positions.append(np.array([0, start_distance ]))
      starting_orientations.append(orientations[a])

   # reset the environment (use multi-agent environment)
   state = env.reset(starting_position, starting_orientation)

   # run the all the episodes at the same time
   for t in range(duration * fs):
      output_angle = policy.act(state)
      state, reward, done = env.step(output_angle.cpu().detach().numpy(), 10)
      if done:
            break 
   end_distances = env.end_distances

   approach_scores = []
   for a in range(n_episodes):
      approach_score = 1 - (end_distances[a] / start_distances[a])
      approach_scores.append(approach_score)



   return np.mean(approach_scores)



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
f_sens_range = [1.]#[1., 1.3, 1.6] #np.arange(0.3, 3, 0.3)
f_motor_range = np.arange(0, 5, 1)
connection_scaling_factor = np.arange(0.25, 5, 0.25)
n_connectivity_variations = 100
asymmetry_range = np.arange(0, 1, 0.1)
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

print(environment)
print(n_oscillators)


# execute the grid search
perform_grid_search(n_episodes, sensitivity_range, k_range, f_sens_range, f_motor_range, n_connectivity_variations, connection_scaling_factor, asymmetry_range,  n_oscillators, environment, stimulus_ratio)

# open the grid search results 
with open(r"GridSearchResults_21_9.pickle", "rb") as input_file:
   grid_results = pickle.load(input_file)
evaluate_grid_search(grid_results)

# visualize the grid search results
other_parameters = {"sensitivity": 5, "k": 5., "f_sens": 1., "f_motor": 1., "a_sens": 0.5, "a_ips": 1.5, "a_con": 0.4, "a_motor": 0.5, "scaling_factor": 0.1}
#visualize_grid_search(grid_results, "a_con", "a_ips", other_parameters)

# evaluate a specific combination of parameters
sensitivity = 5.
k = 5.
f_sens = 0.
f_motor = 0.
a_sens = 0.5
a_ips = 1.5
a_con = 0.4
a_motor = 0.5
n_episodes = 1

scaling = 0.1
#for scaling in np.linspace(0.05, 2, 10):

#evaluate_parameters_concatenated(n_episodes, sensitivity, k, f_sens, f_motor, a_sens*scaling, a_ips*scaling, a_con*scaling, a_motor*scaling, False)


starting_orientation = random.uniform(-np.pi, np.pi)
print(starting_orientation)
starting_position = np.array([0, -random.randrange(95, 105)])

env = Environment(fs, duration, stimulus_position, stimulus_decay_rate,
   stimulus_scale, sensitivity, starting_position, starting_orientation, movement_speed, agent_radius, agent_eye_angle, delta_orientation)

evaluate_parameters(env, n_episodes, k, f_sens, f_motor, a_sens*scaling, a_ips*scaling, a_con*scaling, a_motor*scaling, True)
