#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : training_RL.py
# description     : train agents using the REINFORCE method
# author          : Nicolas Coucke
# date            : 2022-08-16
# version         : 1
# usage           : python training_RL.py
# notes           : install the packages with "pip install -r requirements.txt"
# python_version  : 3.9.2
# ==============================================================================

import matplotlib.pyplot as plt
from collections import deque
import numpy as np
from utils import symmetric_matrix, eucl_distance
from agent import Agent

from environment import Environment
from agent_RL import Gina

import time
from matplotlib import animation
import tkinter as tk
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

fs = 20
duration = 30
stimulus_position = [0, 0]
stimulus_decay_rate = 0.01
stimulus_scale = 10
stimulus_sensitivity = 100
starting_position = [0, -100]
starting_orientation = 0
movement_speed = 10
agent_radius = 2
agent_eye_angle = 45
delta_orientation = 0.1*np.pi



def reinforce(policy, optimizer, n_training_episodes, max_t, gamma, print_every):
    # Help us to calculate the score during the training
    scores_deque = deque(maxlen=max_t)
    scores = []
    times = []
    # loop through all episodes
    for i_episode in range(1, n_training_episodes+1):
        saved_log_probs = []
        rewards = []

        #always start from a random position and orientation
        starting_orientation = random.randrange(0, 360)
        #position should always be at 100m distance
        starting_position = np.array([0, -random.randrange(10, 100)])

       # starting_orientation = 0
       # starting_position =  np.array([0, -100])

        state = env.reset(starting_position, starting_orientation)

        # Line complete the whole episode
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done = env.step(action)
            rewards.append(reward)
            if done:
                break 
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))
        times.append(env.time)
        ## calculate the discounts
        discounts = [gamma**i for i in range(len(rewards)+1)]

        ## calculate sum of discounted rewards
        R = sum([a*b for a,b in zip(discounts, rewards)])
       # R = -env.time
        # calculate total loss
        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R) # add minus before log prob
        policy_loss = torch.cat(policy_loss).sum()
      

        # update policy parameters
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
       # print("Episode " + str(i_episode))

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            #print('Episode {}\tAverage Time: {:.2f}'.format(i_episode, np.mean(times)))

            # plot the environment
            N = 1000
            x = np.linspace(-150, 150, N)
            y = np.linspace(-150, 150, N)
            xx, yy = np.meshgrid(x, y)
            xx, yy = np.meshgrid(x, y)
            zz = np.sqrt(xx**2 + yy**2)   
            zs = stimulus_scale * np.exp( - stimulus_decay_rate * np.sqrt(zz))
            plt.contourf(x, y, zs)
            plt.axis('scaled')
            plt.colorbar()

            # plot the trajectory of the agant
            i = 0
            x_prev = env.position_x[0]
            y_prev = env.position_y[0]
            for x, y in zip(env.position_x, env.position_y):
                # later samples are more visible
                a = i/len(env.position_x)
                plt.plot([x_prev, x], [y_prev, y], alpha = a, color = 'red')
                x_prev = x
                y_prev = y
                i+=1
            plt.xlim([-150, 150])
            plt.ylim([-150, 150])
            plt.show()
            plt.plot(scores)
            plt.show()
    return scores, times


env = Environment(fs, duration, stimulus_position, stimulus_decay_rate,
     stimulus_scale, stimulus_sensitivity, starting_position, starting_orientation, movement_speed, agent_radius, agent_eye_angle, delta_orientation)

state_space = 2 # two sensory readings 
hidden_space = 6 # hidden nodes
action_space = 3 # possible actions (left, right, forward)

policy = Gina(device, state_space, action_space, hidden_space).to(device)
learning_rate = 5e-3 #1e-2 #1e-4 
optimizer = optim.Adam(policy.parameters(), learning_rate)


n_training_episodes = 2000
max_t = duration * fs
gamma = 1.0
scores, times = reinforce(policy, optimizer, n_training_episodes, max_t,
                   gamma, 2000)

plt.plot(times)
plt.show()