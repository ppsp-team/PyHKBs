#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : helloworld.py
# description     : run a simple ODE with one oscillator using PyTorch
# author          : Guillaume Dumas
# date            : 2022-07-05
# version         : 1
# usage           : python helloworld.py
# notes           : install the packages with "pip install -r requirements.txt"
# python_version  : 3.9.13
# ==============================================================================

import matplotlib.pyplot as plt
import torch
from torchdiffeq import odeint
import numpy as np;

from scipy.signal import hilbert





# Key parameters of the simulation
duration = 10
fs = 500
fosc = 2
phi0 = 0
t = np.linspace(0, float(duration), int(duration * fs))
phit = 2 * np.pi * t
phit2 = 4 * np.pi * t

signal = np.sin(phit)

signal2 = np.sin(phit2)

phase = np.angle(hilbert(signal))

phase2 = np.angle(hilbert(signal2))
phase_dif = np.angle(np.exp(1j*(phase - phase2)))
phase_dif2 = (phase - phase2)





#plv += np.abs(np.mean(np.exp(1j *(phase_matrix[i, window_start:window_end] - phase_matrix[j, window_start:window_end]))))



plt.plot(t, signal)
plt.plot(t, signal2)

#plt.plot(t, phase)
#plt.plot(t, phase2)
#plt.plot(t, phase_dif)
#plt.plot(t, np.abs(phase_dif))
phase_matrix = np.vstack((phase, phase2))
KOP_in_time = np.abs(np.mean(np.exp(1j * phase_matrix), 0))
KOP_angle_in_time = np.angle(np.mean(np.exp(1j * phase_matrix), 0))

directional_KOP_in_time = np.sign(KOP_angle_in_time)*KOP_in_time
plt.plot(t, KOP_in_time)
plt.plot(t, directional_KOP_in_time)
plt.legend(['signal 1', 'signal 2', 'KOP', 'KOP angle']) # 'phase diff', 'abs phase diff'
#plt.plot(t, phase_dif2)


plt.show()






# Increase precision of float numbers
torch.set_default_dtype(torch.float64)

# Define pi for subsequent trigonometrics
torch.pi = torch.acos(torch.zeros(1)).item() * 2

# Key parameters of the simulation
duration = torch.as_tensor([10.])  # seconds
fs = torch.as_tensor([500])  # Hertz
fosc = torch.as_tensor([10.])  # Hertz
phi0 = torch.as_tensor([0.])  # Radians

# Generate time
t = torch.linspace(start=0, end=float(duration), steps=int(duration * fs))


def osc(t, phi):
    "Simple oscillator equation."
    return torch.as_tensor([2 * torch.pi * float(fosc)])


# Solving the ODE
phit = odeint(osc, phi0, t)

# Visualization of the solution
plt.plot(t, torch.sin(phit))
plt.show()
