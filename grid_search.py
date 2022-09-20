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
from environment import Environment
from visualizations import single_agent_animation, plot_single_agent_run
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



def perform_grid_search(n_episodes, sensitivity_range, k_range, f_sens_range, f_motor_range, a_sens_range, a_ips_range, a_con_range, a_motor_range, connection_scaling_factor):
   n_configurations = 3**9 

   # initialize dataframe to store results
   config = 1 
   grid_results = pd.DataFrame(columns = ["sensitivity", "k", "f_sens", "f_motor", "a_sens", "a_ips", "a_con", "a_motor", "scaling_factor", "performance"])
   starting_orientations = np.linspace(0, 2*np.pi, n_episodes)

   # loop through all parameter combinations
   for sensitivity in sensitivity_range:
      # use the of certain sensitivity for all runs
      env = Environment(fs, duration, stimulus_position, stimulus_decay_rate,
         stimulus_scale, sensitivity, starting_position, starting_orientation, movement_speed, agent_radius, agent_eye_angle, delta_orientation)
      for k in k_range:
         for f_sens in f_sens_range:
            for f_motor in f_motor_range:
               for a_sens in a_sens_range:
                  for a_ips in a_ips_range:
                     for a_con in a_con_range:
                        for a_motor in a_motor_range:
                           for scale in connection_scaling_factor:
                              
                              # test performance of all parameters
                              performance = evaluate_parameters(env, n_episodes, sensitivity, k, f_sens, f_motor, a_sens, a_ips, a_con, a_motor, False)

                              # save the average score for this configuration
                              configuration_result = pd.Series(data=[sensitivity, k, f_sens, f_motor, a_sens, a_ips, a_con, a_motor, scale, performance],  index= grid_results.columns, name = config)
                              grid_results = grid_results.append(configuration_result)
                              print( "configuration " + str(config) + " of " + str(n_configurations))
                              config+=1
   # save the results
   with open(r"GridSearchResults.pickle", "wb") as output_file: 
      pickle.dump(grid_results, output_file, protocol=pickle.HIGHEST_PROTOCOL)
   print("done")                          
   return grid_results
         

def visualize_grid_search(grid_results, x_axis, y_axis, other_parameters):
   # extract requested values, 
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
      starting_orientation = random.uniform(0, 2*np.pi)
      starting_orientation = 0
      starting_position = np.array([0, -random.randrange(95, 105)])
      starting_position = np.array([0, -100])

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
sensitivity_range = [1., 5., 10.,] #np.arange(0.5, 10, 0.5)
k_range = [1., 2., 5.,] #np.arange(1, 10, 1)
f_sens_range = [1., 1.3, 1.6] #np.arange(0.3, 3, 0.3)
f_motor_range = [1.] #np.arange(0.3, 3, 0.3)
a_sens_range = [1., 2.5, 5.] # np.arange(0.5, 5, 0.5)
a_ips_range = [1., 2.5, 5.]  #np.arange(0.5, 5, 0.5)
a_con_range =  [1., 2.5, 5.]   #np.arange(0.1, 1, 0.1)
a_motor_range = [1., 2.5, 5.]   #  np.arange(0.5, 5, 0.5)


k_range = [1.]
f_sens_range = [1.]
f_motor_range = [1.]
a_sens_range = [0.02]
a_ips_range = [0.02]
a_con_range =  [0.08]
a_motor_range = [0.08]


connection_scaling_factor = [0.01, 0.1, 1.]

n_episodes = 1
# execute the grid search
#grid_results = perform_grid_search(n_episodes, sensitivity_range, k_range, f_sens_range, f_motor_range, a_sens_range, a_ips_range, a_con_range, a_motor_range, connection_scaling_factor)

# open the grid search results 
#with open(r"GridSearchResults.pickle", "rb") as input_file:
   # grid_results = pickle.load(input_file)

# visualize the grid search results
#other_parameters = {"sensitivity": 1, "k": 1., "f_sens": 1., "f_motor": 1., "a_sens": 0.02, "a_ips": 0.02, "a_con": 0.08, "a_motor": 0.08, "scaling_factor": 1.}
#visualize_grid_search(grid_results, "sensitivity", "scaling_factor", other_parameters)

# evaluate a specific combination of parameters
sensitivity = 15.
k = 5.
f_sens = 0.
f_motor = 0.
a_sens = 0.05
a_ips = 0.05
a_con = 1.25
a_motor = 0.5
n_episodes = 1

scaling = 1
#for scaling in np.linspace(0.05, 2, 10):
evaluate_parameters(n_episodes, sensitivity, k, f_sens, f_motor, a_sens*scaling, a_ips*scaling, a_con*scaling, a_motor*scaling, True)
