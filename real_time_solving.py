#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : real_time_solving.py
# description     : DRAFT SCRIPT to solve the system of equations at every timestep 
#                   so that it can deal with inputs
#                   comparison between torchdiffeq and custom RK function
# author          : Nicolas Coucke
# date            : 2022-07-12
# version         : 1
# usage           : python real_time_solving.py
# notes           : install the packages with "pip install -r requirements.txt"
# python_version  : 3.9.2
# ==============================================================================

import matplotlib.pyplot as plt
import torch
import numpy as np
from torchdiffeq import odeint
from utils import symmetric_matrix_torch
import time

# general parameters of simiulation
duration = torch.as_tensor([5.])  # seconds
fs = torch.as_tensor([100])  # Hertz
# oscillator parameters
number_of_oscillators = 5
initial_phases = torch.tensor([0., 0.2, 0.6, 1.2, torch.pi])
frequencies = torch.tensor([1.2, 1.3, 1.4, 1.5, 1.6])
phase_coupling_matrix = symmetric_matrix_torch(0.4)  # phase coupling
anti_phase_coupling_matrix = symmetric_matrix_torch(0.3)  # anti-phase coupling

#initialize variables
phases = torch.as_tensor(np.zeros(5,))
phase_differences = torch.as_tensor(np.zeros(5,))


def single_oscillator(oscillator_number, phases):
    "The phase of a oscillator i is modified by being connected to the other oscillators j by means of the HKB equations"
    oscillator_phase = phases[oscillator_number] # phase of oscillator i
    phase_difference = torch.as_tensor([2 * torch.pi * frequencies[oscillator_number]])
    for other_oscillator_number in range(number_of_oscillators): # loop through all other oscillators
        if other_oscillator_number != oscillator_number:
            # get the phase and coupling variables for oscillator j
            phase_coupling = phase_coupling_matrix[oscillator_number,
                other_oscillator_number]
            anti_phase_coupling = anti_phase_coupling_matrix[oscillator_number, 
                other_oscillator_number]
            other_oscillator_phase = phases[other_oscillator_number]
            # phase change according to HKB equation
            phase_difference += torch.as_tensor([- phase_coupling * torch.sin(oscillator_phase
                - other_oscillator_phase) - anti_phase_coupling * torch.sin(2 * (oscillator_phase - other_oscillator_phase))])
    return phase_difference


def oscillator_system(t, phases):
    "System of N mutually influencing oscillators"
    for oscillator_number in range(number_of_oscillators):
        phase_differences[oscillator_number] = single_oscillator(oscillator_number,phases)
    return phase_differences


def runge_kutta_HKB(phases):
    "Integrate the HKB equation system using RK4"
    k1 = oscillator_system(t, phases) * (1/fs)
    k2 = oscillator_system(t, phases + 0.5 * k1) * (1/fs)
    k3 = oscillator_system(t, phases + 0.5 * k2)  * (1/fs)
    k4 = oscillator_system(t, phases + k3) * (1/fs)
    new_phases = phases + (1/6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return new_phases


# Generate time and intialize variables for simulation
start_rk_time = time.time()  # start timer for RK method
t = torch.linspace(start=0, end=float(duration), steps=int(duration * fs))
phase_time_series = torch.as_tensor(np.zeros((len(t), number_of_oscillators)))
phase_time_series[0, :] = initial_phases

# Execute Runge-Kutta at each timepoint
phases = initial_phases
for i in range(len(t[:-1])):
    new_phases = runge_kutta_HKB(phases)
    phase_time_series[i, :] = new_phases
    phases = new_phases

end_rk_time = time.time()
print(end_rk_time-start_rk_time)  # time elapsed with RK method

# visualize result of simulation with RK
for oscillator_number in range(number_of_oscillators):
    plt.plot(t, torch.sin(phase_time_series[:, oscillator_number]))
plt.show()


# solve with torchdiffeq
# initialize variables for simulation
start_torch_time = time.time()  # start timer for torch method
t = torch.linspace(start=0, end=float(duration), steps=int(duration * fs))
phase_time_series = torch.as_tensor(np.zeros((len(t), number_of_oscillators))) 
phase_time_series[0, :] = initial_phases

for i in range(len(t[:-1])):
    phase_time = odeint(oscillator_system, initial_phases, torch.as_tensor(t[i: i + 2]))
    phase_time_series[i, :] = phase_time[1]
    initial_phases = phase_time[1]

end_torch_time = time.time()
print(end_torch_time-start_torch_time)  # time elapsed with pytorch method

# visualize result of simulation
for oscillator_number in range(number_of_oscillators):
    plt.plot(t, torch.sin(phase_time_series[:, oscillator_number]))
plt.show()
