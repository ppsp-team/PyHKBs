#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : network_of_oscillators.py
# description     : model a network of oscillators with a HKB coupling matrix
# author          : Nicolas Coucke
# date            : 2022-07-11
# version         : 1
# usage           : python network_of_oscillators.py
# notes           : install the packages with "pip install -r requirements.txt"
# python_version  : 3.9.2
# ==============================================================================

import matplotlib.pyplot as plt
import torch
import numpy as np
from torchdiffeq import odeint
from utils import symmetric_matrix_torch

# parameters of simiulation
duration = torch.as_tensor([10.])  # seconds
fs = torch.as_tensor([500])  # Hertz

number_of_oscillators = 5
initial_phases = torch.tensor([0., 0.2, 0.6, 1.2, torch.pi])
frequencies = torch.tensor([1.2, 1.3, 1.4, 1.5, 1.6])

phase_coupling_matrix = symmetric_matrix_torch(0.4)  # phase coupling

anti_phase_coupling_matrix = symmetric_matrix_torch(0.3)  # anti-phase coupling

phases = torch.as_tensor(np.zeros(5,))
phase_differences = torch.as_tensor(np.zeros(5,))


def single_oscillator(oscillator_number, phases):
    "describes how the phase of a single oscillator is modified by being connected to the other oscillators by means of the HKB equations"
    oscillator_phase = phases[oscillator_number]
    phase_difference = torch.as_tensor([2 * torch.pi * frequencies[oscillator_number]])
    for other_oscillator_number in range(number_of_oscillators): #loop through all other oscillators
        if other_oscillator_number != oscillator_number:
            # get the phase and coupling variables for each oscillators
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
        phase_differences[oscillator_number] = single_oscillator(oscillator_number, phases)
    return phase_differences


# Generate time
t = torch.linspace(start=0, end=float(duration), steps=int(duration * fs))

# solve system of equations
phase_time_series = odeint(oscillator_system, initial_phases, t)

# visualize individual oscillations
for oscillator_number in range(number_of_oscillators):
    plt.plot(t, torch.sin(phase_time_series[:, oscillator_number]))
plt.legend()
plt.show()


