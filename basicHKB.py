#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : basicHKB.py
# description     : Illustrate basic HKB properties and solve equation using Pytorch
# author          : Nicolas Coucke
# date            : 2022-07-06
# version         : 1
# usage           : python basicHKB.py
# notes           : install the packages with "pip install -r requirements.txt"
# python_version  : 3.9.2
# ==============================================================================

import matplotlib.pyplot as plt
import torch
import numpy as np
from torchdiffeq import odeint

# We will first plot the phase space of the equation

# Generate parameters to plot phase space
frequency_difference = 0.  # Hertz, difference between intrinsic frequencies; this can modify the "strength" of an attractor/repeller
phase_coupling = 5 # strength of in-phase coupling
anti_phase_coupling = 2  # strength of anti-phase coupling
# higher values of b with respect to a lead to a larger stable region of anti-phase coupling 
# (try setting a to 5 and varying b to 0,1,2,5)


# Generate phase range  in which to plot equation
phase_range = np.linspace(-np.pi, 2 * np.pi, 540)


# define HKB equations without Pytorch
def HKBextended(phase): 
    "Extended HKB equation."
    return frequency_difference - phase_coupling * np.sin(phase) - 2 * anti_phase_coupling * np.sin(2 * phase)


# Visualization of the phase space and find fixed points

# plot zero line
plt.plot(phase_range, np.zeros((len(phase_range))), color = 'black', linewidth = 1)

# plot phase space
plt.plot(phase_range, HKBextended(phase_range))#, color = 'lightblue')

# plot point attractors
for i in range(1, len(phase_range - 1)):
    # indicate attractors in green
    if (HKBextended(phase_range[i]) < 0) & (HKBextended(phase_range[i - 1]) > 0):
        plt.scatter(phase_range[i], HKBextended(phase_range[i]), color='green', s = 100)
    # indicate repellers in red
    elif (HKBextended(phase_range[i]) > 0) & (HKBextended(phase_range[i - 1]) < 0):
        plt.scatter(phase_range[i], HKBextended(phase_range[i]), color='red', s = 100)
plt.xlabel("phase " + r'$\phi$')
plt.ylabel("phase change " + r'$\dot{\phi}$')
plt.xlim([-np.pi, 2*np.pi])
plt.show()


# #######Simulation#########
# now we will solve the HKB equations with pytorch

# Increase precision of float numbers
torch.set_default_dtype(torch.float64)

# Define pi for subsequent trigonometrics
torch.pi = torch.acos(torch.zeros(1)).item() * 2

# Key parameters of the simulation in pytorch format
duration = torch.as_tensor([2.])  # seconds
fs = torch.as_tensor([500])  # Hertz
fosc = torch.as_tensor([10.])  # Hertz
initial_phase = torch.as_tensor([torch.pi + 0.1])  # Radians
phase_coupling_torch = torch.as_tensor([phase_coupling])  # phase coupling
anti_phase_coupling_torch = torch.as_tensor([anti_phase_coupling])  # anti-phase coupling
frequency_difference_torch = torch.as_tensor([0.])

# Generate time
t = torch.linspace(start=0, end=float(duration), steps=int(duration * fs))

# Define HKB equations in Pytorch
def HKBextended_torch(t, phase):
    "Extended HKB equation."
    return torch.as_tensor([frequency_difference - phase_coupling_torch *
     torch.sin(phase) - 2 * anti_phase_coupling_torch * torch.sin(2 * phase)])


# Solve the ODE for different initial values of phi0
for i in np.linspace(0, 2 * np.pi, 20):
    initial_phase = torch.as_tensor([i])
    phase_time_series = odeint(HKBextended_torch, initial_phase, t)
    plt.plot(t, phase_time_series, color='blue')
plt.xlabel("time (s)")
plt.ylabel("phase " + r'$\phi$')

# Visualization of the phase trajectory
plt.ylim([0, 2 * np.pi])
plt.show()
