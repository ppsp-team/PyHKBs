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


def calculate_wPLI(phase_1, phase_2):
   delta_phase = phase_1 - phase_2
   Im = np.imag(np.exp(1j*(delta_phase)))
   numer = np.abs(np.mean(np.abs(Im) * np.sign(Im)))
   denom = np.mean(np.abs(Im))

   if denom == 0:
      denom = 1
   return numer / denom


def calculate_average_inter_agent_PLV(phase_matrices, window_length, window_step, fs):
   # calculate windowed PLV
   window_start = 0
   window_end = int(window_start + window_length)
   window_step = int(window_step)
   simulation_length =int(np.size(phase_matrices[0], 1))
   plv_in_time = []
   plv_5_in_time = []
   interval_times = []
   _n_oscillators = np.size(phase_matrices[0], 0)
   n_agents = len(phase_matrices)
   agent_combinations = n_agents * (n_agents - 1) / 2

   # for the whole duration of the trial
   while (window_start + window_length) < simulation_length:
      interval_times.append((window_start + window_length/2)/fs)
      plv = 0
      counter = 0
      plv_5 = 0


      # loop through all the agent pairs
      for a_i in range(n_agents):
         for a_j in range(a_i+1, n_agents):
            # for all the oscillators
            for i in range(_n_oscillators):
               #plv += np.abs(np.mean(np.exp(1j *(phase_matrices[a_i][i, window_start:window_end] - phase_matrices[a_j][i, window_start:window_end]))))
               wpli = calculate_wPLI(phase_matrices[a_i][i, window_start:window_end] , phase_matrices[a_j][i, window_start:window_end])
               plv +=  wpli #np.abs(np.mean(np.exp(1j *(phase_matrices[a_i][i, window_start:window_end] - phase_matrices[a_j][i, window_start:window_end]))))
               if i == 4: # if you have a fifth oscillator
                  this_plv_5 = np.abs(np.mean(np.exp(1j *(phase_matrices[a_i][i, window_start:window_end] - phase_matrices[a_j][i, window_start:window_end]))))
                  plv_5 += this_plv_5
                  plv += this_plv_5
               counter += 1 
      window_start += window_step
      window_end += window_step
               

      plv_in_time.append(plv / counter)
      plv_5_in_time.append(plv_5 / (counter/_n_oscillators) )

   mean_plv = np.mean(plv_in_time)
   mean_plv_5 = np.mean(plv_5_in_time)


   return plv_in_time, interval_times, mean_plv, plv_5_in_time, mean_plv_5


def calculate_average_PLV(phase_matrices, window_length, window_step, fs):
   # calculate windowed PLV
   window_start = 0
   window_end = int(window_start + window_length)
   window_step = int(window_step)
   simulation_length =int(np.size(phase_matrices[0], 1))
   plv_in_time = []
   plv_5_in_time = []
   interval_times = []
   _n_oscillators = np.size(phase_matrices[0], 0)
   n_agents = len(phase_matrices)
   oscillator_combinations = _n_oscillators * (_n_oscillators - 1) / 2
  


   while (window_start + window_length) < simulation_length:
      interval_times.append((window_start + window_length/2)/fs)
      plv = 0
      counter = 0
      for a in range(len(phase_matrices)):
         phase_matrix = phase_matrices[a]
         for i in range(_n_oscillators):
            for j in range(i+1, _n_oscillators): # i+1 because dont want connection of oscillator with itself
               #plv += np.abs(np.mean(np.exp(1j *(phase_matrix[i, window_start:window_end] - phase_matrix[j, window_start:window_end]))))
               wpli = calculate_wPLI(phase_matrices[a][i, window_start:window_end] , phase_matrices[a][j, window_start:window_end])
               plv+= wpli
               counter += 1
      window_start += window_step
      window_end += window_step
      plv_in_time.append(plv / counter)
   mean_plv = np.mean(plv_in_time)

   return plv_in_time, interval_times, mean_plv






# define variables for environment
fs = 50 # Hertz
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
starting_orientation = 0.2
starting_distances = [100]#np.linspace(95, 105, )
starting_orientations = [0.2] # np.linspace(-np.pi/2, np.pi/2, 5)
environment = "double_stimulus"



a_sens = 0.
a_ips_left = 0.
a_ips_right= 0.
a_con_left = 0.5
a_con_right = 0.5
a_motor = 0.
scale = 2.
stimulus_sensitivity = 2
social_sensitivity = 0
social_weight_decay_rate = 0.01

f_sens = 5.
f_motor = 5.
k = 2
a_soc_sens_left = 0.
a_soc_sens_right = 0.
a_soc_motor_left = 0.8
a_soc_motor_right = 0.8
n_oscillators = 5
flavour = 'social'
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
    agent_stimulus_decay_rate = 0.1
    env = Social_stimulus_environment(fs, duration, stimulus_positions, stimulus_decay_rate,
     stimulus_scale, stimulus_sensitivity, movement_speed, agent_radius, agent_eye_angle, delta_orientation, agent_stimulus_scale, agent_stimulus_decay_rate, stimulus_ratio, n_agents)

runs = evaluate_parameters_social(env, device, fs, duration, starting_distances, starting_orientations, k, intrinsic_frequencies, coupling_weights, social_sensitivity, social_weight_decay_rate, n_oscillators, flavour, n_agents, False)

run = runs[0]
positions_x = run["x position"]
positions_y = run["y position"]
for a in range(n_agents):
    plt.plot(positions_x[a], positions_y[a])
plt.show()

plot_multi_agent_run(stimulus_ratio, stimulus_decay_rate, stimulus_scale, positions_x, positions_y , n_agents)


# choose one of the players
phase_matrix = run["phases"][0]
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

