import numpy as np
from utils import symmetric_matrix, eucl_distance, initiate_coupling_weights
from environment import Environment, Social_environment
from simulations import evaluate_parameters
from visualizations import single_agent_animation, plot_single_agent_run, plot_single_agent_multiple_trajectories
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
fs = 250# Hertz
duration = 30 # Seconds
stimulus_positions = [np.array([-100, 0]), np.array([100,0])] # m, m
stimulus_decay_rate = 0.02 # in the environment
stimulus_scale = 1 # in the environment
stimulus_sensitivity = 1 # of the agent
movement_speed = 10 #m/s
delta_orientation = 0.1*np.pi # rad/s turning speed # not used anymore here
stimulus_ratio = 0.
agent_radius = 2.5
agent_eye_angle = 0.5 * np.pi # 90 degrees
starting_position = [0, -100] 
starting_orientation = -0.25*np.pi
starting_distances = [100]#np.linspace(95, 105, )
starting_orientations = [-0.] # np.linspace(-np.pi/2, np.pi/2, 5)
environment = "double_stimulus"



a_sens = 0.
a_ips_left = 0.
a_ips_right= 0.
a_con_left = 0.8
a_con_right = 0.8
a_motor = 0.

scale = 4

stimulus_sensitivity = 5
f_sens = 5.
f_motor = 5.
k = 2
a_soc_sens_left = 0.
a_soc_sens_right = 0.
a_soc_motor_left = 0.2
a_soc_motor_right = 0.2
n_oscillators = 5

if n_oscillators == 4:
   coupling_weights = scale * np.array([ a_sens, a_ips_left, a_ips_right, a_con_left, a_con_right, a_motor])
   intrinsic_frequencies = np.array([f_sens, f_motor])
else:
   coupling_weights = 0.75 * scale * np.array([ a_sens, a_ips_left, a_ips_right, a_con_left, a_con_right, a_motor,
                    a_soc_sens_left, a_soc_sens_right, a_soc_motor_left, a_soc_motor_right])
   intrinsic_frequencies = np.array([f_sens, f_motor, f_motor])

#for scale in np.linspace(0.1, 0.5, 5):
#coupling_weights = initiate_coupling_weights(scale, 0., False, 4)[0]
print(coupling_weights)


env = Environment(fs, duration, stimulus_positions, stimulus_ratio, stimulus_decay_rate,
   stimulus_scale, stimulus_sensitivity, starting_position, starting_orientation, movement_speed, agent_radius, agent_eye_angle, delta_orientation)

runs = evaluate_parameters(env, device, duration, fs, starting_distances, starting_orientations, k, intrinsic_frequencies, coupling_weights, n_oscillators)

# choose one of the runs
run = runs[0]
print(run["end time"])

# for this function we have to adjust the data structure
# plot_single_agent_multiple_trajectories(all_positions_x, all_positions_y, stimulus_scale, stimulus_decay_rate, environment, stimulus_ratio)



x_position = run["x position"]
y_position = run["y position"]
phase_differences = run["phase differences"]
input_values = run["input values"]
angles = run["output angle"]
actions = run["orientation"]
phases = run["phases"]


plot_single_agent_run(f_sens, f_motor, coupling_weights, k, x_position, y_position, phase_differences, input_values, angles, actions, stimulus_scale, stimulus_ratio, stimulus_decay_rate)
single_agent_animation(x_position, y_position, phases, phase_differences, stimulus_scale, stimulus_decay_rate,  stimulus_ratio, duration, fs)


phase_matrix = run["phases"]

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




