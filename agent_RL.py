#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : agent_RL.py
# description     : contains neural agents that can be trained in pytorch
# author          : Nicolas Coucke
# date            : 2022-08-16
# version         : 1
# usage           : use within training_RL.py
# notes           : install the packages with "pip install -r requirements.txt"
# python_version  : 3.9.2
# ==============================================================================

import matplotlib.pyplot as plt
import numpy as np
from utils import symmetric_matrix, eucl_distance
from agent import Agent
import time
from matplotlib import animation
import tkinter as tk
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical




class Gina(nn.Module):
    """
    A neural network RL agent to be trained with policy gradient methods
    """

    def __init__(self, device):
        super(Gina, self).__init__()
        s_size = 2 # observation for left and right eye
        a_size = 3 # right, left, forward
        h_size = 6 # hidden nodes
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)
        self.device = device

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


class Guido(nn.Module):
    """
    An RL agent with HKB equations instead of neural network layers

    Arguments
    ---------
    frequency: np or torch array of length 2
        should be [sensory_frequency, motor_frequency]

    phase_coupling: np or torch array of length 4
        should be [a_sensory, a_ipsilateral, a_contralateral, a_motor]

    k: int or float
        the proportion of phase coupling vs anti-phase coupling
    """

    def __init__(self, device, fs,  frequency = np.array([]), phase_coupling = np.array([]), k = -1):
        # also execute base initialization of nn.Module
        super().__init__() 

        # initialize the frequencies
        if frequency.size == 0:
            # random values if no input argument
            self.frequency = nn.Parameter(torch.Tensor(2))
            self.frequency = nn.init.uniform_(self.frequency, a = 0.3, b = 3)
        else:
            self.frequency = nn.Parameter(torch.tensor(frequency))


        # initialize the 'a' phase coupling values
        if phase_coupling.size == 0:
            self.phase_coupling = nn.Parameter(torch.Tensor(4))
            self.phase_coupling = nn.init.uniform_(self.phase_coupling, a = 0.1, b = 5)
        else:
            self.phase_coupling = nn.Parameter(torch.tensor(phase_coupling))

        # initialize k
        if k == -1:
            # random values
            self.k = nn.Parameter(torch.Tensor(1))
            self.k = nn.init.uniform_(self.k, a = 1, b = 10)
        else:
            self.k = nn.Parameter(torch.tensor([float(k)]))
            

        # layer to calculate the next phase of the oscillators
        self.full_step_layer = FullStepLayer(fs, self.frequency, self.phase_coupling, self.k)

        # layer to get the probabilities for each action
        self.softmax = nn.Softmax(dim=None)
        self.device = device

        # initialize phases
        self.phases = torch.tensor([0., 0., 0., 0.])
        

    def forward(self, input):
        # transform  input to two eyes into a vector 
        # with one value per oscillator
        input = torch.squeeze(input)
        new_input = torch.zeros(4)
        new_input[0] = input[0]
        new_input[1] = input[1]
        
       
        # execute forward step of the oscillator phases
        phase_difference = self.full_step_layer(new_input, self.phases)
        self.phases += phase_difference 
        output_angle = self.phases[3] - self.phases[2]
        output_angle = output_angle % 2 * torch.pi 
        a = torch.sqrt(torch.tensor(1/(6 * torch.pi))) # corresponds to a cartoid with area of 1

        # define probabilities for taking actions according to the phase angle
        # positive values make right turn more probable
        #prob_right = #torch.heaviside(torch.sin(output_angle / 4), torch.tensor([0.]))

        # make probabilities by defining a cartoid centered around - 90 deg
        prob_right = a * (1 - torch.sin(output_angle))

        # centered around 90 deg
        prob_left = a * (1 - torch.sin( - output_angle))

        # acentered around 0 deg
        prob_forward = a * (1 - torch.cos(torch.pi - output_angle))

        # transform output probabilities with softmax
        probs = torch.tensor([prob_right, prob_left, prob_forward])
        probs = torch.unsqueeze(probs, 0)
        probs.requires_grad = True
        return F.softmax(probs, dim=1)


    def act(self, state):
        # do forward pass (update update oscillator phases)
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.forward(state).cpu()
        # pick an action
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def reset(self, chosen_phases):
        self.phases = chosen_phases
        


class FullStepLayer(nn.Module):
    """
    Layer that, given the current phases and inputs, calculates 
    the phases of the oscillators at the next timestep 
    by solving the system of differential equations using the Runge-Kutta method
    """
    def __init__(self, fs, frequency, phase_coupling, k):
        super().__init__()
        self.fs = fs
        # initialize the runge kutta step layer that calculates the phase differences for each oscillator
        # we will pass the input and phases through this layer 4 times in order to execute the runge kutta method
        self.runge_kutta_step = RungeKuttaStepLayer(fs, frequency, phase_coupling, k)

    def forward(self, x, phases):
        k1 = self.runge_kutta_step.forward(x, phases) * (1/self.fs)
        k2 = self.runge_kutta_step.forward(x, phases + 0.5 * k1) * (1/self.fs)
        k3 = self.runge_kutta_step.forward(x, phases + 0.5 * k2)  * (1/self.fs)
        k4 = self.runge_kutta_step.forward(x, phases + k3) * (1/self.fs)
        phase_differences = (1/6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return phase_differences


class RungeKuttaStepLayer(nn.Module):
    """
    Pytorch layer that that calculates the phase change for each oscillator using the full HKB equations
    This layer is used sevaral times when using the Runge Kutta method
    """
    def __init__(self, fs, frequency, phase_coupling, k):
        super().__init__()
        
        
        self.frequency_array = torch.tensor([frequency[0], frequency[0], frequency[1], frequency[1]])

        # create var for each coupling value for clarity
        a_sens = phase_coupling[0]
        a_ips = phase_coupling[1]
        a_con = phase_coupling[2]
        a_motor = phase_coupling[3]

        # we initialize couping matrix with the chosen weight
        self.phase_coupling_matrix = torch.tensor([[0, a_sens, a_ips, a_con],
                                            [a_sens, 0, a_con, a_ips],
                                            [a_ips, a_con, 0, a_motor],
                                            [a_con, a_ips, a_motor, 0]])

        # the anti phase weights are proportional to the phase weights
        self.anti_phase_coupling_matrix =  self.phase_coupling_matrix / k

        # these layers will calculate the mutual influence of the oscillators
        self.phase_layer = MutualInfluenceLayer(self.phase_coupling_matrix, torch.tensor([1]))
        self.anti_phase_layer = MutualInfluenceLayer(self.anti_phase_coupling_matrix, torch.tensor([2]))

        self.fs = fs
        


    def forward(self, x, phases):
        # get the phase change for each oscillator
        # x is the sensory input to each oscillator
        x =  self.frequency_array + x - self.phase_layer(phases) - self.anti_phase_layer(phases)
        return x


class MutualInfluenceLayer(nn.Module):
    """
    Pytorch layer that calculates the phase change for each oscillator in a network of mutually influencing oscillators

    """
    def __init__(self, coupling_matrix, phase_multiplyer):
        """
        Arguments:
        ---------
        coupling_matrix: torch.Tensor of dim (n_oscillators, n_oscillators)
            contains the coupling weight for each pair of oscillators

        phase_multiplyer: scalar 
            1 for in-phase coupling
            2 for anti-phase coupling

        """
        super().__init__()
        self.weights = torch.flatten(coupling_matrix)
        self.phase_multiplyer = phase_multiplyer
        self.num_oscillators = coupling_matrix.size(dim=0)


    def forward(self, phases):
        """
        Arguments:
        ---------
        phases: torch.Tensor of dim n_oscillators
            contains the phase of each oscillator in the previous timestep

        Returns:
        ---------
        weighted_sums: torch.Tensor of dim n_oscillators
            contains the phase change for each oscillators resulting from its (in/anti) phase coupling with the other oscillators

        """
        phases_i = torch.unsqueeze(phases, 1)
        phases_i = torch.flatten(phases_i.repeat(1, self.num_oscillators))
        phases_j = torch.squeeze(phases.repeat(1, self.num_oscillators), 0)
        phase_differences = self.phase_multiplyer * (phases_i - phases_j)
        weighted_differences = self.weights * torch.sin(phase_differences)
        weighted_differences = torch.reshape(weighted_differences, (self.num_oscillators, self.num_oscillators))
        weighted_sums = torch.sum(weighted_differences, dim=1)
        return weighted_sums
