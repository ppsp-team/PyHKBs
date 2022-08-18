#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : agent_RL.py
# description     : contains neural agents that can be trained in pytorch
# author          : Nicolas Coucke
# date            : 2022-08-16
# version         : 1
# usage           : python single_agent_visualization
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class Gina(nn.Module):
    """
    A neural network RL agent to be trained with policy gradient methods
    """

    def __init__(self, device, s_size, a_size, h_size):
        super(Gina, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
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
    A neural network RL agent to be trained with policy gradient methods
    """

    def __init__(self, device, s_size, a_size, h_size):
        super(Gina, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
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


