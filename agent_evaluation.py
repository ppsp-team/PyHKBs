#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : agent_RL.py
# description     : allows evaluating single agent runs
# author          : Nicolas Coucke
# date            : 2022-10-16
# version         : 1
# usage           : python agent_evaluation.py
# notes           : install the packages with "pip install -r requirements.txt"
# python_version  : 3.9.2
# ==============================================================================



import numpy as np
from utils import symmetric_matrix, eucl_distance, initiate_coupling_weights
from environment import Environment, Social_environment
from simulations import evaluate_parameters
from visualizations import single_agent_animation, plot_single_agent_run, plot_single_agent_multiple_trajectories, plot_single_agent_run_simplified
from agent_RL import Gina, Guido, MultipleGuidos

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
import tkinter as tk
import random
import pickle

from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def calculate_KOP(phase_matrix):
   """
    Calculate the Kuramoto order parameter of phase time series in matrix 

    Arguments: 
    ---------
    phase_matrix: matrix of dims (n_oscillators, time)

    Returns:
    --------
    KOP_in_time: array of length time

    KOP_std: scalar standard deviation of KOP
    """
   
   KOP_in_time = np.abs(np.mean(np.exp(1j * phase_matrix), 0))
   KOP_std = np.std(KOP_in_time)
   return KOP_in_time, KOP_std


def calculate_average_PLV(phase_matrix, window_length, window_step):
   """
    Calculate average of windowed pairwize PLV values between signals

    Arguments: 
    ---------
    phase_matrix: matrix of dims (n_oscillators, time)
    window_length: scalar (samples)
    window_step: scalar (samples)
    

    Returns:
    --------
    plv_in_time: array of length time/window_step
    interval_times: array of length time/window_step 
    mean_plv: scalar 
   """
   window_start = 0
   window_end = window_start + window_length
   simulation_length =int(np.size(phase_matrix, 1))
   plv_in_time = []
   interval_times = []
   oscillator_combinations = n_oscillators * (n_oscillators - 1) / 2

   while (window_start + window_length) < simulation_length:
      interval_times.append((window_start + window_length/2))
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


def calculate_separate_PLV_in_time(phase_matrix, window_length, window_step, fs):
   """
    Calculate average of windowed pairwize PLV values between signals

    Arguments: 
    ---------
    phase_matrix: matrix of dims (n_oscillators, time)
    window_length: scalar (samples)
    window_step: scalar (samples)
    fs: scalar (sampling frequency)
    

    Returns:
    --------
    plv_time_list: list of pairwize PLV values
    interval_times: array of length time/window_step 
   """
   window_start = 0
   window_end = window_start + window_length
   simulation_length =int(np.size(phase_matrix, 1))
   plv_in_time = []
   interval_times = []
   oscillator_combinations = n_oscillators * (n_oscillators - 1) / 2

   plv_time_list = []
   plv_time_list.append([])
   plv_time_list.append([])
   plv_time_list.append([])

   while (window_start + window_length) < simulation_length:
      interval_times.append((window_start + window_length/2)/fs)
      plv = 0
      counter = 0
      
      # left eye to right motor 
      i = 0
      j = 3
      plv = np.abs(np.mean(np.exp(1j *(phase_matrix[i, window_start:window_end] - phase_matrix[j, window_start:window_end]))))
      plv_time_list[0].append(plv)

      # right eye to left  motor 
      i = 1
      j = 2
      plv = np.abs(np.mean(np.exp(1j *(phase_matrix[i, window_start:window_end] - phase_matrix[j, window_start:window_end]))))
      plv_time_list[1].append(plv)

      # left motor to right motor
      i = 2
      j = 3
      plv = np.abs(np.mean(np.exp(1j *(phase_matrix[i, window_start:window_end] - phase_matrix[j, window_start:window_end]))))
      plv_time_list[2].append(plv)


      window_start += window_step
      window_end += window_step
      counter += 1
   return plv_time_list, interval_times





# define variables for environment
fs = 100#100# Hertz
duration = 30 # Seconds
stimulus_positions = [np.array([-100, 0]), np.array([100,0])] # m, m
stimulus_decay_rate = 0.02 # in the environment


stimulus_scale = 1.0 # in the environment
stimulus_sensitivity = 1 # of the agent
movement_speed = 10 #m/s
delta_orientation = 0.1*np.pi # rad/s turning speed # not used anymore here
stimulus_ratio = 0.99
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
a_motor = 0.3

scale = 2

stimulus_sensitivity = 4
f_sens = 5.
f_motor = 5.
k = 2
a_soc_sens_left = 0.
a_soc_sens_right = 0.
a_soc_motor_left = 0.2
a_soc_motor_right = 0.2
n_oscillators = 4

if n_oscillators == 4:
   coupling_weights = scale * np.array([ a_sens, a_ips_left, a_ips_right, a_con_left, a_con_right, a_motor])
   intrinsic_frequencies = np.array([f_sens, f_motor])
else:
   coupling_weights = 0.75 * scale * np.array([ a_sens, a_ips_left, a_ips_right, a_con_left, a_con_right, a_motor,
                    a_soc_sens_left, a_soc_sens_right, a_soc_motor_left, a_soc_motor_right])
   intrinsic_frequencies = np.array([f_sens, f_motor, f_motor])


env = Environment(fs, duration, stimulus_positions, stimulus_ratio, stimulus_decay_rate,
   stimulus_scale, stimulus_sensitivity, starting_position, starting_orientation, movement_speed, agent_radius, agent_eye_angle, delta_orientation)

runs = evaluate_parameters(env, device, duration, fs, starting_distances, starting_orientations, k, intrinsic_frequencies, coupling_weights, n_oscillators, False)

# choose one of the runs
run = runs[0]
print(run["end time"])


x_position = run["x position"]
y_position = run["y position"]
phase_differences = run["phase differences"]
input_values = run["input values"]
angles = run["output angle"]
actions = run["orientation"]
phases = run["phases"]

plot_single_agent_run_simplified(x_position, y_position, stimulus_scale, stimulus_ratio, stimulus_decay_rate)
#plot_single_agent_run(f_sens, f_motor, coupling_weights, k, x_position, y_position, phase_differences, input_values, angles, actions, stimulus_scale, stimulus_ratio, stimulus_decay_rate)
#single_agent_animation(x_position, y_position, phases, phase_differences, stimulus_scale, stimulus_decay_rate,  stimulus_ratio, duration, fs, True)


phase_matrix = run["phases"]
window_length = int(fs)
window_step = int(fs/10)
plv_time_list, interval_times = calculate_separate_PLV_in_time(phase_matrix, window_length, window_step, fs)

fig, ax = plt.subplots()

line_1, = ax.plot(interval_times, plv_time_list[0])
line_2, = ax.plot(interval_times, plv_time_list[1])
line_3, = ax.plot(interval_times, plv_time_list[2])


ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

ax.legend([line_1, line_2, line_3], ['sens-motor left', 'sens-motor right', 'motor-motor'])
plt.ylabel('PLV')
plt.xlabel('time')
plt.show()
   





