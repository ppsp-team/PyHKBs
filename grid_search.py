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

import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def perform_grid_search(n_episodes, sensitivity_range, k_range, f_sens_range, f_motor_range, a_sens_range, a_ips_range, a_con_range, a_motor_range, connection_scaling_factor):
   n_configurations = len(sensitivity_range)*len(k_range)*len(f_sens_range)*len(f_motor_range)*len(a_sens_range)*len(a_ips_range)*len(a_con_range)*len(a_motor_range)*len(connection_scaling_factor)

   # initialize dataframe to store results
   config = 1 
   grid_results = pd.DataFrame(columns = ["sensitivity", "k", "f_sens", "f_motor", "a_sens", "a_ips", "a_con", "a_motor", "scaling_factor", "performance"])
   starting_orientations = np.linspace(0, 2*np.pi, n_episodes)

   # loop through all parameter combinations
   for sensitivity in sensitivity_range:
      # use the of certain sensitivity for all runs
      env = Environment(fs, duration, stimulus_position, stimulus_decay_rate,
         stimulus_scale, sensitivity, starting_position, starting_orientation, movement_speed, agent_radius, agent_eye_angle, delta_orientation)
      #env = Social_environment(fs, duration, stimulus_position, stimulus_decay_rate,
        #stimulus_scale, stimulus_sensitivity, movement_speed, agent_radius, agent_eye_angle, delta_orientation, stimulus_ratio)
      for k in k_range:
         for f_sens in f_sens_range:
            for f_motor in f_motor_range:
               for a_sens in a_sens_range:
                  for a_ips in a_ips_range:
                     for a_con in a_con_range:
                        for a_motor in a_motor_range:
                           for scale in connection_scaling_factor:
                              
                              # test performance of all parameters
                              performance = evaluate_parameters(env, n_episodes, k, f_sens, f_motor, a_sens*scale, a_ips*scale, a_con*scale, a_motor*scale, False)
                              # save the average score for this configuration
                              configuration_result = pd.Series(data=[sensitivity, k, f_sens, f_motor, a_sens, a_ips, a_con, a_motor, scale, performance],  index= grid_results.columns, name = config)
                              grid_results = grid_results.append(configuration_result)
                              print( "configuration " + str(config) + " of " + str(n_configurations))
                              config+=1
         # save the results every few runs
         with open(r"GridSearchResults.pickle", "wb") as output_file: 
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



def evaluate_parameters(env, n_episodes, k, f_sens, f_motor, a_sens, a_ips, a_con, a_motor, plot):

   
   frequency = np.array([f_sens, f_motor])
   phase_coupling = np.array([a_sens, a_con, a_ips, a_motor])
   # create agent with these parameters
   policy = Guido(device, fs, frequency, phase_coupling, k).to(device)

   approach_scores = []
   # do ten episodes for each parameter combination
   for i in range(n_episodes):
        # reset the environment 
      starting_orientation = random.uniform(-0.5*np.pi, 0.5*np.pi)
      starting_position = np.array([0, -random.randrange(50, 100)])
      #starting_position = np.array([0, -100])

      state = env.reset(starting_position, starting_orientation)

      # reset Guido
      policy.reset(torch.tensor([0., 0., 0, 0.]))

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
         # anim = single_agent_animation(env.position_x, env.position_y, phases, phase_differences, stimulus_scale, stimulus_decay_rate, duration, fs)
         #anim.save('GuidoSimulation.gif')

   return np.mean(approach_scores)
      


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
   orientations = np.random.uniform(- 0.25 * np.pi, 0.25 * np.pi, n_episodes)

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
fs = 20 # Hertz
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
sensitivity_range = [5.]#[1., 5., 10.] #np.arange(1, 20, 1)
k_range = [5.] #[1., 2., 5.,] #np.arange(1, 10, 1)
f_sens_range = [1.]#[1., 1.3, 1.6] #np.arange(0.3, 3, 0.3)
f_motor_range = [1.] #[1.] #np.arange(0.3, 3, 0.3)
a_sens_range = np.arange(0.5, 5, 0.5)
a_ips_range = np.arange(0.5, 5, 0.5)
a_con_range =  np.arange(0.1, 1, 0.1)
a_motor_range = np.arange(0.5, 5, 0.5)

connection_scaling_factor = [0.1]#[0.01, 0.1, 1.]

n_episodes = 50
# execute the grid search
grid_results = perform_grid_search(n_episodes, sensitivity_range, k_range, f_sens_range, f_motor_range, a_sens_range, a_ips_range, a_con_range, a_motor_range, connection_scaling_factor)

# open the grid search results 
with open(r"GridSearchResults_21_9.pickle", "rb") as input_file:
   grid_results = pickle.load(input_file)
evaluate_grid_search(grid_results)

# visualize the grid search results
other_parameters = {"sensitivity": 5, "k": 5., "f_sens": 1., "f_motor": 1., "a_sens": 1., "a_ips": 1., "a_con": 5, "a_motor": 5, "scaling_factor": 0.1}
#visualize_grid_search(grid_results, "a_con", "a_ips", other_parameters)

# evaluate a specific combination of parameters
sensitivity = 5.
k = 5.
f_sens = 1.
f_motor = 1.
a_sens = 1.
a_ips = 5.
a_con = 1.5
a_motor = 5.
n_episodes = 1

scaling = 0.1
#for scaling in np.linspace(0.05, 2, 10):

#evaluate_parameters_concatenated(n_episodes, sensitivity, k, f_sens, f_motor, a_sens*scaling, a_ips*scaling, a_con*scaling, a_motor*scaling, False)


starting_orientation = random.uniform(-np.pi, np.pi)
print(starting_orientation)
starting_position = np.array([0, -random.randrange(95, 105)])

env = Environment(fs, duration, stimulus_position, stimulus_decay_rate,
   stimulus_scale, sensitivity, starting_position, starting_orientation, movement_speed, agent_radius, agent_eye_angle, delta_orientation)

#evaluate_parameters(env, n_episodes, k, f_sens, f_motor, a_sens*scaling, a_ips*scaling, a_con*scaling, a_motor*scaling, True)
