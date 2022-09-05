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



def perform_grid_search(env, n_episodes, sensitivity_range, k_range, f_sens_range, f_motor_range, a_sens_range, a_ips_range, a_con_range, a_motor_range, connection_scaling_factor):
   n_configurations = 3**9 
   config = 1
   grid_results = pd.DataFrame(columns = ["sensitivity", "k", "f_sens", "f_motor", "a_sens", "a_ips", "a_con", "a_motor", "scaling_factor", "performance"])
   starting_orientations = np.linspace(0, 2*np.pi, n_episodes)
   print(starting_orientations)
   # loop through all parameter combinations
   for sensitivity in sensitivity_range:
      for k in k_range:
         for f_sens in f_sens_range:
            for f_motor in f_motor_range:
               for a_sens in a_sens_range:
                  for a_ips in a_ips_range:
                     for a_con in a_con_range:
                        for a_motor in a_motor_range:
                           for scale in connection_scaling_factor:
                              frequency = np.array([f_sens, f_motor])
                              phase_coupling = np.array([a_sens, a_con, a_ips, a_motor]) * scale

                              # create agent with these parameters
                              policy = Guido(device, fs, frequency, phase_coupling, k).to(device)

                              # do ten episodes per parameter configuration
                              approach_scores = []
                              for episode in range(n_episodes):
                                 # reset the environment 

                                 starting_orientation = random.uniform(0, 2*np.pi)
                                 starting_position = np.array([0, -random.randrange(95, 105)])
                                 state = env.reset(starting_position, starting_orientation)

                                 # reset Guido
                                 policy.reset(torch.tensor([0., 0., 0., 0.]))
                                 
                                 # Complete the whole episode
                                 start_distance = env.distance
                                 for t in range(duration * fs):
                                    #action, log_prob = policy.act(state)
                                    action, log_prob, output_angle = policy.act(state)
                                    #state, reward, done = env.step(action, 10)
                                    state, reward, done = env.step(output_angle.cpu().detach().numpy(), 10)
                                    if done:
                                          break 
                                 end_distance = env.distance

                                 approach_score = 1 - (end_distance / start_distance)
                                 approach_scores.append(approach_score)
                              
                              # save the average score for each episode
                              performance = np.mean(approach_score)
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
        # print(key)
         #print(grid_results[key] == other_parameters[key])
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



def evaluate_parameters(env, n_episodes, sensitivity, k, f_sens, f_motor, a_sens, a_ips, a_con, a_motor):

   frequency = np.array([f_sens, f_motor])
   phase_coupling = np.array([a_sens, a_con, a_ips, a_motor])
   # create agent with these parameters
   policy = Guido(device, fs, frequency, phase_coupling, k).to(device)
   approach_scores = []
   # do ten episodes for each parameter combination
   fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
   for i in range(n_episodes):
        # reset the environment 
      starting_orientation = random.uniform(0, 2*np.pi)
      starting_orientation = np.pi /4
      starting_position = np.array([0, -random.randrange(95, 105)])
      starting_position = np.array([0, -100])


      state = env.reset(starting_position, starting_orientation)

      # reset Guido
      #policy.reset(torch.tensor([0., np.pi, 0, np.pi]))
      #policy.reset(torch.tensor(np.pi*np.random.rand(4)))
      policy.reset(torch.tensor([0., 0., 0, 0.]))

      # Complete the whole episode
      start_distance = env.distance
      input_values = np.zeros((2, fs * duration))
      phase_differences = np.zeros((4, fs * duration))
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
         actions.append(env.orientation)
         angles.append(policy.output_angle.cpu().detach().numpy())
      end_distance = env.distance
      approach_score = 1 - (end_distance / start_distance)
      approach_scores.append(approach_score)


      # plot the trajectory of the agant in the environment
      i = 0
      x_prev = env.position_x[0]
      y_prev = env.position_y[0]
      for x, y in zip(env.position_x, env.position_y):
         # later samples are more visible
         a = 0.1 + 0.5*i/len(env.position_x)
         ax1.plot([x_prev, x], [y_prev, y], alpha = a, color = 'red')
         x_prev = x
         y_prev = y
         i+=1
      ax1.set_xlim([-150, 150])
      ax1.set_ylim([-150, 150])

   # plot influence of the input over the oscillator changes
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

# define variables for environment
fs = 30 # Hertz
duration = 30 # Seconds
stimulus_position = [0, 0] # m, m
stimulus_decay_rate = 0.02 # in the environment
stimulus_scale = 10 # in the environment
stimulus_sensitivity = 2 # of the agent
starting_position = [0, -100] 
starting_orientation = 0 
movement_speed = 20 #m/s
delta_orientation = 0.2*np.pi # rad/s turning speed
agent_radius = 5
agent_eye_angle = 45


# use the same environment for all runs 
env = Environment(fs, duration, stimulus_position, stimulus_decay_rate,
     stimulus_scale, stimulus_sensitivity, starting_position, starting_orientation, movement_speed, agent_radius, agent_eye_angle, delta_orientation)


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
a_sens_range = [1.]
a_ips_range = [1.]
a_con_range =  [1.]
a_motor_range = [1.]


connection_scaling_factor = [0.01, 0.1, 1.]

n_episodes = 10
# execute the grid search
grid_results = perform_grid_search(env, n_episodes, sensitivity_range, k_range, f_sens_range, f_motor_range, a_sens_range, a_ips_range, a_con_range, a_motor_range, connection_scaling_factor)



# open the grid search results 
with open(r"GridSearchResults.pickle", "rb") as input_file:
    grid_results = pickle.load(input_file)

# visualize the grid search results
other_parameters = {"sensitivity": 1, "k": 1., "f_sens": 1., "f_motor": 1., "a_sens": 1., "a_ips": 1., "a_con": 1., "a_motor": 1., "scaling_factor": 1.}
visualize_grid_search(grid_results, "sensitivity", "scaling_factor", other_parameters)

# evaluate a specific combination of parameters
sensitivity = 2
k = 5/2.
f_sens = 1.
f_motor = 1.
a_sens = 0.25
a_ips = 0.25
a_con = 0.5
a_motor = 0.25
n_episodes = 1
evaluate_parameters(env, n_episodes, sensitivity, k, f_sens, f_motor, a_sens, a_ips, a_con, a_motor)
