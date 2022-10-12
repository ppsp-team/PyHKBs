#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : training_RL.py
# description     : train and evaluate agents using the REINFORCE method
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

from environment_RL import Environment
#from environment import Environment
from agent_RL import Gina, Guido
import wandb
import socket
from pathlib import Path
import sys

import time
from matplotlib import animation
import tkinter as tk
import random
import os
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv


import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

print(torch.version.cuda)
print(torch.cuda.device_count())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

agent = Guido
batch_size = 10



def train_one_epoch(agent, optimizer, env, max_t, gamma, print_every, batch_size):
    # make some empty lists for logging.
    batch_obs = []          # for observations
    batch_acts = []         # for actions
    batch_weights = []      # for R(tau) weighting in policy gradient
    batch_rets = []         # for measuring episode returns
    batch_lens = []         # for measuring episode lengths

    # reset episode-specific variables
    obs = env.reset()       # first obs comes from starting distribution
    done = False            # signal from environment that episode is over
    ep_rews = []            # list for rewards accrued throughout ep

    # render first episode of each epoch
    finished_rendering_this_epoch = False

    # collect experience by acting in the environment with current policy
    while True:

        # rendering
      #  if (not finished_rendering_this_epoch) and render:
         #   env.render()

        # save obs
        batch_obs.append(obs.copy())

        # act in the environment
        action, logp  = agent.act(torch.as_tensor(obs, dtype=torch.float32))
        obs, rew, done, _ = env.step(action)

        # save action, reward
        batch_acts.append(action)
        ep_rews.append(rew)

        if done:
            # if episode is over, record info about episode
            ep_ret, ep_len = sum(ep_rews), len(ep_rews)
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)

            # the weight for each logprob(a|s) is R(tau)
            batch_weights += [ep_ret] * ep_len

            # reset episode-specific variables
            obs, done, ep_rews = env.reset(), False, []

            # won't render again this epoch
            finished_rendering_this_epoch = True

            # end experience loop if we have enough of it
            if len(batch_obs) > batch_size:
                break

    # take a single policy gradient update step
    optimizer.zero_grad()

    # make loss function whose gradient, for the right data, is policy gradient
    weights = torch.as_tensor(batch_weights, dtype=torch.float32)
    batch_loss = -(logp * weights).mean()
    batch_loss.backward()
    optimizer.step()
    return batch_loss, batch_rets, batch_lens



def train_agent():



# define variables for environment
fs = 30 # Hertz
duration = 30 # Seconds
stimulus_position = [0, 0] # m, m
stimulus_decay_rate = 0.02 # in the environment
stimulus_scale = 10 # in the environment
stimulus_sensitivity = 1 # of the agent
starting_position = [0, -100] 
starting_orientation = 0 
movement_speed = 10
delta_orientation = 0.3*np.pi # turning speed
agent_radius = 2
agent_eye_angle = 45




def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Environment":
                env = Environment(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + " environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(
            all_args.n_rollout_threads)])


#envs = make_train_env(fs, duration, stimulus_position, stimulus_decay_rate,
    #stimulus_scale, stimulus_sensitivity, starting_position, starting_orientation, movement_speed, agent_radius, agent_eye_angle, delta_orientation)

# create an environment object in which the agent will be trained
env = Environment(fs, duration, stimulus_position, stimulus_decay_rate,
     stimulus_scale, stimulus_sensitivity, starting_position, starting_orientation, movement_speed, agent_radius, agent_eye_angle, delta_orientation)

# create an agent as policy 
#policy = Gina(device).to(device)
sensitivity = 10
k = 5
f_sens = 2.
f_motor = 2
a_sens = 0.1
a_ips = 0.5
a_con = 5
a_motor = 0.2
n_episodes = 10

# initialize guido with good variables
frequency = np.array([f_sens, f_motor])
phase_coupling = np.array([a_sens, a_con, a_ips, a_motor])
policy = Guido(device, fs, frequency , phase_coupling, k).to(device)
#policy = Gina(device).to(device)

# variables for training
learning_rate = 0.01 #1e-2 #1e-4 
optimizer = optim.Adam(policy.parameters(), learning_rate)
n_training_episodes = 10
max_t = duration * fs
gamma = 0.99




# start training
#scores, times = reinforce(policy, optimizer, n_training_episodes, max_t,
#                  gamma, 10)



def reinforce(policy, optimizer, n_training_episodes, max_t, gamma, print_every):
    """
    Optimizes the parameters of a policy using the REINFORCE method

    Arguments:
    ----------
    policy: object
        An agent with a differentiable Pytorch architecture

    optimizer: object
        PyTorch optimization object

    n_training_episodes: int

    max_t: int
        maximum steps within an episode

    gamma: float
        discount rate for rewards

    Returns:
    ----------
    scores: list
        the score for each episode

    times: list
        the times taken for each episode (when variable)   

    """
    # Help us to calculate the score during the training
    scores_deque = deque(maxlen=max_t)
    scores = []
    times = []

    # loop through all episodes
    for i_episode in range(1, n_training_episodes+1):
        food_size = 5 #+ 30 * (1 - (i_episode / n_training_episodes))
        saved_log_probs = []
        rewards = []

        # always start from a random position and orientation
        starting_orientation = random.uniform(0, 2*np.pi)
        starting_orientation = 0
        # start on a random point on the line going from 10 to 100m from center
        starting_position = np.array([0, -random.randrange(95, 105)])

        # the environment keeps track of the agent's position
        state = env.reset(starting_position, starting_orientation)

        # Complete the whole episode
        start_distance = env.distance
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done = env.step(action, food_size)
            rewards.append(reward)
            if done:
                break 
        end_distance = env.distance
        scores_deque.append(sum(rewards))
        #scores.append(sum(rewards))
        times.append(env.time)
        print(env.time)
        print(env.position)
      
        
        discounted_rewards = []
        for t in range(len(rewards)):
            Gt = 0 
            pw = 0
            for r in rewards[t:]:
                Gt = Gt + gamma**pw * r
                pw = pw + 1
                discounted_rewards.append(Gt)
        
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards

                # reward for coming close
        approach_score = 1 - (end_distance / start_distance)
        scores.append(approach_score)

        # calculate total loss 
        # we actually want to do gradient ascent
        # but it's easier to do descent in pytorch so we just add a minus
        policy_gradient = []
        for log_prob, Gt in zip(saved_log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * approach_score) #Gt)
           
         
        # calculate the discounts
        #discounts = [gamma**i for i in range(len(rewards)+1)]   
        # calculate sum of discounted rewards
        # R = sum([a*b for a,b in zip(discounts, rewards)])
        # policy_loss = []
        # i = 0
        #for log_prob in saved_log_probs:
        #    policy_loss.append(-log_prob * discounts[i]*rewards[i]) 
        # policy_loss = torch.cat(policy_loss).sum()
      
        # update policy parameters
        optimizer.zero_grad()
        #policy_loss.backward()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        optimizer.step()
        
        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            #print('Episode {}\tAverage Time: {:.2f}'.format(i_episode, np.mean(times)))

            # plot the environment with stimulus concentration
            N = 1000
            x = np.linspace(-150, 150, N)
            y = np.linspace(-150, 150, N)
            xx, yy = np.meshgrid(x, y)
            xx, yy = np.meshgrid(x, y)
            zz = np.sqrt(xx**2 + yy**2)   
            zs = stimulus_scale * np.exp( - stimulus_decay_rate * zz)
            plt.contourf(x, y, zs)
            plt.axis('scaled')
            plt.colorbar()

            # plot the trajectory of the agant in the environment
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

            # also show how the scores have evolved
            plt.plot(scores)
            plt.show()
    return scores, times

