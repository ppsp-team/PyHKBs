#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : two_oscillators.py
# description     : Explicitly model two oscillators of which the connection is governed by the HKB equations
# author          : Nicolas Coucke
# date            : 2022-07-07
# version         : 1
# usage           : python two_oscillators.py
# notes           : install the packages with "pip install -r requirements.txt"
# python_version  : 3.9.2
# ==============================================================================

import matplotlib.pyplot as plt
import torch
import numpy as np
from torchdiffeq import odeint

# parameters of simiulation
duration = torch.as_tensor([10.])  # seconds
fs = torch.as_tensor([500])  # Hertz

initial_phase_1 = torch.as_tensor([torch.pi])  # Radians, initial phase of oscillator 1
initial_phase_2 = torch.as_tensor([0])  # Radians, initial phase of oscillator 2

frequency_1 = torch.as_tensor([1.2])  # Hertz, intrinsic frequency of oscillator 1
frequency_2 = torch.as_tensor([1.6])  # Hertz, intrinsic frequency of oscillator 2

phase_coupling = torch.as_tensor([0.4])  # phase coupling
anti_phase_coupling = torch.as_tensor([0.3])  # anti-phase coupling


def HKBextended(t, phase):
    "System of two oscillators"
    phase1 = torch.remainder(phase[0], 2 * torch.pi)  # clamp phases between 0 and 2pi
    phase2 = torch.remainder(phase[1], 2 * torch.pi)
    phase_difference_1 = torch.as_tensor([2 * torch.pi * frequency_1 - phase_coupling * torch.sin(phase1 - phase2) - anti_phase_coupling * torch.sin(2 * (phase1 - phase2))])
    phase_difference_2 = torch.as_tensor([2 * torch.pi * frequency_2 - phase_coupling * torch.sin(phase2 - phase1) - anti_phase_coupling * torch.sin(2 * (phase2 - phase1))])
    return ( phase_difference_1, phase_difference_2)


# Generate time
t = torch.linspace(start=0, end=float(duration), steps=int(duration * fs))

# solve system of equations
phase_time_series = odeint(HKBextended, (initial_phase_1, initial_phase_2), t)

# visualize individual oscillations
plt.plot(t, torch.sin(phase_time_series[0]))
plt.plot(t, torch.sin(phase_time_series[1]))
plt.show()

# plot phase difference
plt.plot(t, torch.remainder(phase_time_series[0] - phase_time_series[1], 2 * torch.pi))
plt.ylim([0, 2 * np.pi])
plt.show()

# plot intrinsic frequencies (find out why doesn't work)
phase_time_series_1 = phase_time_series[0].detach().cpu().numpy()  # convert tensors to numpy
phase_time_series_2 = phase_time_series[1].detach().cpu().numpy()
phase_time_series_1 = phase_time_series_1.flatten()
phase_time_series_2 = phase_time_series_2.flatten()
time = t.detach().cpu().numpy()
plt.plot(time[:-1], np.diff(phase_time_series_1) / (2 * np.pi))  # calculate frequency as the normalized derivative of phases
plt.plot(time[:-1], np.diff(phase_time_series_2) / (2 * np.pi))
# plt.ylim([0,2.5])
# plt.plot(t, np.diff(phit[1])/(2*np.pi))
plt.show()
