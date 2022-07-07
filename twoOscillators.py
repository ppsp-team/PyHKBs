#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : twoOscillators.py
# description     : Explicitly model two oscillators of which the connection is governed by the HKB equations
# author          : Nicolas Coucke
# date            : 2022-07-07
# version         : 1
# usage           : python twoOscillators.py
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

phi1_0 = torch.as_tensor([torch.pi])  # Radians, initial phase of oscillator 1
phi2_0 = torch.as_tensor([0])  # Radians, initial phase of oscillator 2

f1 = torch.as_tensor([1.2])  # Hertz, intrinsic frequency of oscillator 1
f2 = torch.as_tensor([1.6])  # Hertz, intrinsic frequency of oscillator 2

a = torch.as_tensor([0.4])  # phase coupling
b = torch.as_tensor([0.3])  # anti-phase coupling


def HKBextended(t, phi):
    "System of two oscillators"
    phi1 = torch.remainder(phi[0], 2 * torch.pi)  # clamp phases between 0 and 2pi
    phi2 = torch.remainder(phi[1], 2 * torch.pi)
    dphi1 = torch.as_tensor([2 * torch.pi * f1 - a * torch.sin(phi1 - phi2) - b * torch.sin(2 * (phi1 - phi2))])
    dphi2 = torch.as_tensor([2 * torch.pi * f2 - a * torch.sin(phi2 - phi2) - b * torch.sin(2 * (phi2 - phi1))])
    return (dphi1, dphi2)


# Generate time
t = torch.linspace(start=0, end=float(duration), steps=int(duration * fs))

# solve system of equations
phit = odeint(HKBextended, (phi1_0, phi2_0), t)

# visualize individual oscillations
plt.plot(t, torch.sin(phit[0]))
plt.plot(t, torch.sin(phit[1]))
plt.show()

# plot phase difference
plt.plot(t, torch.remainder(phit[0] - phit[1], 2 * torch.pi))
plt.ylim([0, 2 * np.pi])
plt.show()

# plot intrinsic frequencies (find out why doesn't work)
phase1 = phit[0].detach().cpu().numpy()  # convert tensors to numpy
phase2 = phit[1].detach().cpu().numpy()
phase1 = phase1.flatten()
phase2 = phase2.flatten()
time = t.detach().cpu().numpy()
print(np.array(phase1))
plt.plot(time[:-1], np.diff(phase1) / (2 * np.pi))  # calculate frequency as the normalized derivative of phases
plt.plot(time[:-1], np.diff(phase2) / (2 * np.pi))
# plt.ylim([0,2.5])
# plt.plot(t, np.diff(phit[1])/(2*np.pi))
plt.show()
