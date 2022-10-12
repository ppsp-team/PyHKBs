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
from torch.distributions import Distribution




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


class MultipleGuidos(nn.Module):
    """
    A class to parallelize the feedforward run of multiple guido's interacting in the same environment

    Arguments
    --------- 
    size: how many guidos
    
    other parameters are the same as basic Guido class
    
    """

    def __init__(self, device, fs,  frequency = np.array([]), phase_coupling = np.array([]), k = -1, size = 1):
        # also execute base initialization of nn.Module
        super().__init__() 
        self.list_of_guidos = nn.ModuleList([Guido(device, fs, frequency, phase_coupling, k) for i in range(int(size))])

        self.concatenated_guidos = torch.cat(self.list_of_guidos, dim=1)


    def forward(self, input):
        """
        input: concatenated inputs of all the agents

        output: the concatenated output angles of all the agents
        """
        return self.concatenated_guidos(input)

    
    def act(self, state):
        """
        state are the concatenated inputs of the agents

        returns the output angles of all agents
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        output_angles = self.forward(state).cpu()
      
        return output_angles






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

    def __init__(self, device, fs,  frequency = np.array([]), phase_coupling = np.array([]), k = -1, symmetric = True):
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
        self.full_step_layer = FullStepLayer(fs, self.frequency, self.phase_coupling, self.k, symmetric)

        # layer to get the probabilities for each action
        self.softmax = nn.Softmax(dim=None)
        self.device = device

        # initialize phases
        self.phases = torch.tensor([0., 0., 0., 0.])
        self.output_angle = self.phases[3] - self.phases[2]


    def forward(self, input):
        # transform  input to two eyes into a vector 
        # with one value per oscillator
        self.input = torch.squeeze(input)
        new_input = torch.zeros(4)
        new_input[0] = self.input[0]
        new_input[1] = self.input[1]
        
        
        # execute forward step of the oscillator phases
        self.phase_difference = self.full_step_layer(new_input, self.phases)
        self.phases += self.phase_difference 

        output_angle = self.phases[3] - self.phases[2]
        self.output_angle = output_angle # % 2 * torch.pi 
        a = torch.sqrt(torch.tensor(1/(1 * torch.pi))) # corresponds to a cartoid with area of 1

        # define probabilities for taking actions according to the phase angle
        # positive values make right turn more probable

        # make probabilities by defining a cartoid centered around - 90 deg
        prob_left = a * (1 - torch.sin(output_angle))

        # centered around 90 deg
        prob_right = a * (1 - torch.sin( - output_angle))

        # acentered around 0 deg
        prob_forward = a * (1 - torch.cos(torch.pi - output_angle))

       # print(output_angle)
        #prob_right = torch.heaviside(torch.sin(output_angle / 4), torch.tensor([0.]))
       # prob_left = torch.heaviside(- torch.sin(output_angle / 4), torch.tensor([0.]))
       # prob_forward= torch.heaviside(torch.cos( output_angle / 4), torch.tensor([0.]))


        # transform output probabilities with softmax
        probs = torch.tensor([prob_right, prob_left, prob_forward])
       # print(probs)
        probs = torch.unsqueeze(probs, 0)
        probs.requires_grad = True
        return F.softmax(probs, dim=1)


    def act(self, state):
        # do forward pass (update update oscillator phases)
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.forward(state).cpu()
        # pick an action
        m = Categorical(probs)
        action = m.sample() # probabilistic policy
        #action = torch.argmax(probs) # deterministic policy
        return action.item(), m.log_prob(action), self.output_angle

    def reset(self, chosen_phases):
        self.phases = chosen_phases
        


class FullStepLayer(nn.Module):
    """
    Layer that, given the current phases and inputs, calculates 
    the phases of the oscillators at the next timestep 
    by solving the system of differential equations using the Runge-Kutta method
    """
    def __init__(self, fs, frequency, phase_coupling, k, symmetric):
        super().__init__()
        self.fs = fs
        # initialize the runge kutta step layer that calculates the phase differences for each oscillator
        # we will pass the input and phases through this layer 4 times in order to execute the runge kutta method
        self.runge_kutta_step = RungeKuttaStepLayer(fs, frequency, phase_coupling, k, symmetric)

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
    def __init__(self, fs, frequency, phase_coupling, k, symmetric):
        super().__init__()
        
        
        self.frequency_array = torch.tensor([frequency[0], frequency[0], frequency[1], frequency[1]])

        # create var for each coupling value for clarity
        a_sens = phase_coupling[0]
        a_ips = phase_coupling[1]
        a_con = phase_coupling[2]
        a_motor = phase_coupling[3]

        if symmetric == True:
            # we initialize couping matrix with the chosen weight
            self.phase_coupling_matrix = torch.tensor([[0, a_sens, a_ips, a_con],
                                                    [a_sens, 0, a_con, a_ips],
                                                    [a_ips, a_con, 0, a_motor],
                                                    [a_con, a_ips, a_motor, 0]])

            # the anti phase weights are proportional to the phase weights
            self.anti_phase_coupling_matrix =  self.phase_coupling_matrix / k
        else:
            self.phase_coupling_matrix = a_sens * torch.rand(4, 4)
            self.anti_phase_coupling_matrix = a_sens * torch.rand(4, 4) / k

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




class SocialGuido(nn.Module):
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

    def __init__(self, device, fs,  frequency = np.array([]), phase_coupling = np.array([]), k = -1, agent_coupling = -1, n_agents = 1, agent_id = 0):
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
            self.phase_coupling = nn.Parameter(torch.Tensor(5))
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


        # initialize agent coupling w
        if agent_coupling == -1:
            # random values
            self.agent_coupling = nn.Parameter(torch.Tensor(1))
            self.agent_coupling = nn.init.uniform_(self.k, a = 1, b = 10)
        else:
            self.agent_coupling = nn.Parameter(torch.tensor([float(agent_coupling)]))

        # layer to calculate the next phase of the oscillators
        self.social_full_step_layer = SocialFullStepLayer(fs, self.frequency, self.phase_coupling, self.k,  self.agent_coupling, n_agents, agent_id)

        # layer to get the probabilities for each action
        self.softmax = nn.Softmax(dim=None)
        self.device = device

        # initialize phases
        self.phases = torch.tensor([0., 0., 0., 0., 0])
        self.output_angle = self.phases[3] - self.phases[2]

    def forward(self, input, angles, inter_agent_distances):
        # transform  input to two eyes into a vector 
        # with one value per oscillator
        self.input = torch.squeeze(input)
        new_input = torch.zeros(5)
        new_input[0] = self.input[0]
        new_input[1] = self.input[1]

        self.angles = torch.squeeze(angles)
        
        self.inter_agent_distances = torch.squeeze(inter_agent_distances)
        # execute forward step of the oscillator phases
        self.phase_difference = self.social_full_step_layer(new_input, self.phases, self.angles, self.inter_agent_distances)
        self.phases += self.phase_difference 

        output_angle = self.phases[2] - self.phases[3]
        #print(output_angle)

        self.output_angle = output_angle # % 2 * torch.pi 
        #a = torch.sqrt(torch.tensor(1/(6 * torch.pi))) # corresponds to a cartoid with area of 1

        angles = torch.linspace(-torch.pi, torch.pi, 360)
        probs = (1/(2*torch.pi)) * (1 - torch.cos(output_angle - torch.pi - angles))
        # define probabilities for taking actions according to the phase angle
        # positive values make right turn more probable

        probs = torch.unsqueeze(probs, 0)
        probs.requires_grad = True
        return F.softmax(probs, dim=1)


    def act(self, state, angles, inter_agent_distances):
        # do forward pass (update update oscillator phases)
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        angles = torch.from_numpy(angles).float().unsqueeze(0).to(self.device)
        inter_agent_distances = torch.from_numpy(inter_agent_distances).float().unsqueeze(0).to(self.device)
        probs = self.forward(state, angles, inter_agent_distances).cpu()
        # pick an action
        angles = torch.linspace(-torch.pi, torch.pi, 360)
        index = probs.multinomial(num_samples=1, replacement=True)
        action = angles[index]
        log_prob = torch.log(probs[index])
        return action.item(), log_prob, self.output_angle

    def reset(self, chosen_phases):
        self.phases = chosen_phases
        


class SocialFullStepLayer(nn.Module):
    """
    Layer that, given the current phases and inputs, calculates 
    the phases of the oscillators at the next timestep 
    by solving the system of differential equations using the Runge-Kutta method
    """
    def __init__(self, fs, frequency, phase_coupling, k, agent_coupling, n_agents, agent_id):
        super().__init__()
        self.fs = fs
        # initialize the runge kutta step layer that calculates the phase differences for each oscillator
        # we will pass the input and phases through this layer 4 times in order to execute the runge kutta method
        self.runge_kutta_step = SocialRungeKuttaStepLayer(fs, frequency, phase_coupling, k, agent_coupling, n_agents, agent_id)

    def forward(self, x, phases, angles, inter_agent_distances):
        k1 = self.runge_kutta_step.forward(x, phases, angles, inter_agent_distances) * (1/self.fs)
        k2 = self.runge_kutta_step.forward(x, phases + 0.5 * k1, angles,  inter_agent_distances) * (1/self.fs)
        k3 = self.runge_kutta_step.forward(x, phases + 0.5 * k2, angles,  inter_agent_distances)  * (1/self.fs)
        k4 = self.runge_kutta_step.forward(x, phases + k3, angles, inter_agent_distances) * (1/self.fs)
        phase_differences = (1/6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return phase_differences


class SocialRungeKuttaStepLayer(nn.Module):
    """
    Pytorch layer that that calculates the phase change for each oscillator using the full HKB equations
    This layer is used sevaral times when using the Runge Kutta method
    """
    def __init__(self, fs, frequency, phase_coupling, k, agent_coupling, n_agents, agent_id):
        super().__init__()

        self.frequency_array = torch.tensor([frequency[0], frequency[0], frequency[1], frequency[1], frequency[2]])

        # create var for each coupling value for clarity
        a_sens = phase_coupling[0]
        a_ips = phase_coupling[1]
        a_con = phase_coupling[2]
        a_motor = phase_coupling[3]

        a_soc_sens = phase_coupling[4]
        a_soc_motor = phase_coupling[5]

        # we initialize couping matrix with the chosen weight
        self.phase_coupling_matrix = torch.tensor([[0, a_sens, a_ips, a_con, a_soc_sens],
                                            [a_sens, 0, a_con, a_ips, a_soc_sens],
                                            [a_ips, a_con, 0, a_motor, a_soc_motor],
                                            [a_con, a_ips, a_motor, 0, a_soc_motor],
                                            [a_soc_sens, a_soc_sens, a_soc_motor, a_soc_motor, 0]])
        
        n_oscillators = self.phase_coupling_matrix.size(dim=0)

        # the anti phase weights are proportional to the phase weights
        self.anti_phase_coupling_matrix =  self.phase_coupling_matrix / k

        # these layers will calculate the mutual influence of the oscillators
        self.phase_layer = MutualInfluenceLayer(self.phase_coupling_matrix, torch.tensor([1]))
        self.anti_phase_layer = MutualInfluenceLayer(self.anti_phase_coupling_matrix, torch.tensor([2]))

        # these layer will calculate social influence between agents
        self.phase_social_layer = SocialInfluenceLayer(agent_coupling, torch.tensor([1]), agent_id, n_agents, n_oscillators)
        anti_phase_agent_coupling = agent_coupling / k
        self.anti_phase_social_layer = SocialInfluenceLayer(anti_phase_agent_coupling, torch.tensor([2]), agent_id, n_agents, n_oscillators)

        self.fs = fs
        


    def forward(self, input, phases, angles,  inter_agent_distances):
        # get the phase change for each oscillator
        # x is the sensory input to each oscillator
        output =  self.frequency_array + input - self.phase_layer(phases) - self.anti_phase_layer(phases) - self.phase_social_layer(angles,  inter_agent_distances) - self.anti_phase_social_layer(angles,  inter_agent_distances)
        return output





class SocialInfluenceLayer(nn.Module):
    """
    Pytorch layer that calculates the phase change for each oscillator in a network of mutually influencing oscillators

    """
    def __init__(self, coupling_weight, phase_multiplyer, agent_id, n_agents, n_oscillators):
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
        self.coupling_weight = coupling_weight
        self.phase_multiplyer = phase_multiplyer

        # we will use this to get an array of the agent angles from which we can subtract the angles of the other agents
        self.angle_filter = torch.zeros(n_agents, n_agents)
        self.angle_filter[:, agent_id] = torch.ones(n_agents)

        # we only want the social information to enter one of the oscillators
        self.oscillator_filter = torch.zeros(n_oscillators)
        self.oscillator_filter[4] = 1

  

    def forward(self, angles,  inter_agent_distances):
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
        # calcultate the angle differences with the other agents
        self_angles = torch.matmul(self.angle_filter, angles)
        
        angle_differencs = self.phase_multiplyer * (self_angles - angles)

        distances = self.angle_filter * inter_agent_distances

        # create proportional weighted sum
        weighted_differences = self.coupling_weight * torch.exp( - 0.1 * distances ) * torch.sin(angle_differencs)
        weighted_sum = torch.sum(weighted_differences)

        # pass the sum to the correct oscillator
        output = self.oscillator_filter * weighted_sum
       
        return output


