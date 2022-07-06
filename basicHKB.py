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


#Generate parameters to plot phase space
npdOmega = 0. #Hertz, difference between differences in intrinsic frequencies; can modify the "strength" of an attractor/repeller
np_a = 5. #strength of in-phase coupling
np_b = 1. #strength of anti-phase coupling
#higher values of b with respect to a lead to a larger stable region of anti-phase coupling (try setting a to 5 and varying b to 0,1,2,5)


#Generate phase range
phiRange = np.linspace(-np.pi, 2*np.pi, 540)

def npHKBextended(phi):
    "Extended HKB equation."
    return npdOmega-np_a*np.sin(phi)-2*np_b*np.sin(2*phi)


# Visualization of the phase space and find fixed points
for i in range(1,len(phiRange-1)):
    #indicate attractors in green
    print(npHKBextended(phiRange[i]))
    if (npHKBextended(phiRange[i]) < 0) & (npHKBextended(phiRange[i-1]) > 0):
       plt.scatter(phiRange[i],npHKBextended(phiRange[i]), color = 'green')
    #indicate repellers in red
    elif (npHKBextended(phiRange[i]) > 0) & (npHKBextended(phiRange[i-1]) < 0):
       plt.scatter(phiRange[i],npHKBextended(phiRange[i]), color = 'red')
plt.plot(phiRange, npHKBextended(phiRange))
plt.show()


########Simulation#########

# Increase precision of float numbers
torch.set_default_dtype(torch.float64)

# Define pi for subsequent trigonometrics
torch.pi = torch.acos(torch.zeros(1)).item() * 2

# Key parameters of the simulation in pytorch format
duration = torch.as_tensor([2.])  # seconds
fs = torch.as_tensor([500])  # Hertz
fosc = torch.as_tensor([10.])  # Hertz
#phi0 = torch.as_tensor([1.])  # Radians
phi0 = torch.as_tensor([torch.pi+0.1])  # Radians
a = torch.as_tensor([np_a]) #phase coupling
b = torch.as_tensor([np_b]) #anti-phase coupling
dOmega = torch.as_tensor([0.])

# Generate time
t = torch.linspace(start=0, end=float(duration), steps=int(duration * fs))


def HKB(t, phi):
    "Simple HKB equation."
    return torch.as_tensor([-a*torch.sin(phi)-2*b*torch.sin(2*phi)])


def HKBextended(t, phi):
    "Extended HKB equation."
    return torch.as_tensor([dOmega-a*torch.sin(phi)-2*b*torch.sin(2*phi)])

# Solve the ODE for different initial values of phi0
for i in np.linspace(0,2*np.pi, 20):
    phi0 = torch.as_tensor([i])
    phit = odeint(HKBextended, phi0, t)
    plt.plot(t, phit, color = 'blue')

# Visualization of the phase trajectory
plt.ylim([0,2*np.pi])
plt.show()


