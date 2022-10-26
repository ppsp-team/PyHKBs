import numpy as np
from utils import symmetric_matrix, eucl_distance, initiate_coupling_weights
from environment import Environment, Social_environment, Social_stimulus_environment
from simulations import evaluate_parameters_social
from visualizations import plot_multi_agent_run
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



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def calculate_KOP(phase_matrix):
   KOP_in_time = np.abs(np.mean(np.exp(1j * phase_matrix), 0))
   KOP_std = np.std(KOP_in_time)
   return KOP_in_time, KOP_std

def calculate_average_PLV(phase_matrix, window_length, window_step):
   # calculate windowed PLV
   window_start = 0
   window_end = window_start + window_length
   simulation_length =int(np.size(phase_matrix, 1))
   plv_in_time = []
   interval_times = []
   oscillator_combinations = n_oscillators * (n_oscillators - 1) / 2
   print(n_oscillators)
   print(oscillator_combinations)
   print(np.size(phase_matrix, 0))

   while (window_start + window_length) < simulation_length:
      interval_times.append(window_start + window_length/2)
      plv = 0
      counter = 0
      for i in range(n_oscillators):
         for j in range(i+1, n_oscillators): # i+1 because dont want connection of oscillator with itself
            plv += np.abs(np.mean(np.exp(1j *(phase_matrix[i, window_start:window_end] - phase_matrix[j, window_start:window_end]))))
            window_start += window_step
            window_end += window_step
            counter += 1
      plv_in_time.append(plv / oscillator_combinations)
   mean_plv = np.mean(plv_in_time)

   return plv_in_time, interval_times, mean_plv






# define variables for environment
fs = 100# Hertz
duration = 30 # Seconds
stimulus_positions = [np.array([-100, 0]), np.array([100,0])] # m, m
stimulus_decay_rate = 0.02 # in the environment
stimulus_scale = 0.5 # in the environment
stimulus_sensitivity = 1 # of the agent
movement_speed = 10 #m/s
delta_orientation = 0.1*np.pi # rad/s turning speed # not used anymore here
stimulus_ratio = 0.8
agent_radius = 2.5
agent_eye_angle = 0.5 * np.pi # 45 degrees
starting_position = [0, -100] 
starting_orientation = -0.25*np.pi
starting_distances = [100]#np.linspace(95, 105, )
starting_orientations = [0.2] # np.linspace(-np.pi/2, np.pi/2, 5)
environment = "double_stimulus"



a_sens = 0.
a_ips_left = 0.
a_ips_right= 0.
a_con_left = 0.5
a_con_right = 0.5
a_motor = 0.05
scale = 2.
stimulus_sensitivity = 5
social_sensitivity = 0
social_weight_decay_rate = 0.01

f_sens = 5.
f_motor = 5.
k = 5
a_soc_sens_left = 0.
a_soc_sens_right = 0.
a_soc_motor_left = 0.5
a_soc_motor_right = 0.5
n_oscillators = 5
flavour = 'eco'
n_agents = 10

if n_oscillators == 4:
   coupling_weights = scale * np.array([ a_sens, a_ips_left, a_ips_right, a_con_left, a_con_right, a_motor])
   intrinsic_frequencies = np.array([f_sens, f_motor])
else:
   coupling_weights = 0.75 * scale * np.array([ a_sens, a_ips_left, a_ips_right, a_con_left, a_con_right, a_motor,
                    a_soc_sens_left, a_soc_sens_right, a_soc_motor_left, a_soc_motor_right])
   intrinsic_frequencies = np.array([f_sens, f_motor, f_motor])
print(coupling_weights)

if flavour == "social":
    env = Social_environment(fs, duration, stimulus_positions, stimulus_decay_rate,
        stimulus_scale, stimulus_sensitivity, movement_speed, agent_radius, agent_eye_angle, delta_orientation, stimulus_ratio, n_agents)

elif flavour == "eco":
    agent_stimulus_scale = 0.002
    agent_stimulus_decay_rate = 0.1
    env = Social_stimulus_environment(fs, duration, stimulus_positions, stimulus_decay_rate,
     stimulus_scale, stimulus_sensitivity, movement_speed, agent_radius, agent_eye_angle, delta_orientation, agent_stimulus_scale, agent_stimulus_decay_rate, stimulus_ratio, n_agents)

all_approach_scores, all_positions_x, all_positions_y, all_input_values, all_phases, all_phase_differences, all_angles, all_actions = evaluate_parameters_social(env, device, fs, duration, starting_distances, starting_orientations, k, intrinsic_frequencies, coupling_weights, social_sensitivity, social_weight_decay_rate, n_oscillators, flavour, n_agents, False)

positions_x = all_positions_x[0] # take one run
positions_y = all_positions_y[0]
for a in range(n_agents):
    plt.plot(positions_x[a], positions_y[a])
plt.show()

print(all_approach_scores)
plot_multi_agent_run(stimulus_ratio, stimulus_decay_rate, stimulus_scale, positions_x, positions_y , n_agents)






# choose one of the runs
phase_matrix = all_phases[0][0]

# calculate the KOP
KOP_in_time, KOP_std = calculate_KOP(phase_matrix)

# and the PLV
window_length = int(fs)
window_step = int(fs/10)
plv_in_time, interval_times, mean_plv = calculate_average_PLV(phase_matrix, window_length, window_step)


print(mean_plv)
print(KOP_std)
plt.plot(interval_times, plv_in_time)
plt.plot(KOP_in_time)
plt.show()



plot_single_agent_multiple_trajectories(all_positions_x, all_positions_y, stimulus_scale, stimulus_decay_rate, environment, stimulus_ratio)


for x_position, y_position, phases, phase_differences, input_values, angles, actions in zip(all_positions_x, all_positions_y, all_phases, all_phase_differences, all_input_values, all_angles, all_actions):
   plot_single_agent_run(f_sens, f_motor, coupling_weights, k, x_position, y_position, phase_differences, input_values, angles, actions, stimulus_scale, stimulus_ratio, stimulus_decay_rate)
   single_agent_animation(x_position, y_position, phases, phase_differences, stimulus_scale, stimulus_decay_rate,  stimulus_ratio, duration, fs)





# open the grid search results 
with open(r"results/GridSearchResults_single_stimulus_4_1810.pickle", "rb") as input_file:
   grid_results = pickle.load(input_file)

grid_results = average_grid_serach(grid_results)

#show_grid_search_results(grid_results, 10)

max_mean_agent, min_mean_agent, max_min_agent = find_agents(grid_results)
max_mean_agent = max_mean_agent.to_dict()
print(max_mean_agent)

a_sens = max_mean_agent["a_sens"]
a_ips_left = max_mean_agent["a_ips_left"]
a_ips_right= max_mean_agent["a_ips_right"]
a_con_left = max_mean_agent["a_con_left"]
a_con_right = max_mean_agent["a_con_right"]
a_motor = max_mean_agent["a_motor"]
scale = max_mean_agent["scaling_factor"]
stimulus_sensitivity = max_mean_agent["sensitivity"]
f_sens = max_mean_agent["f_sens"]
f_motor = max_mean_agent["f_motor"]
k = max_mean_agent["k"]








def visualize_grid_search(grid_results, x_axis, y_axis, other_parameters):
   """
   plots the results of the grid search on two specified dimension

   Arguments:
   -----------
   grid_results: pandas dataframe

   x_axis: string

   y_axis: string

   other_parameters: dictionary
      the values of the parameters that stay fixed

   
   """
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





