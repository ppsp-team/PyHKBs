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
