from utils import symmetric_matrix, eucl_distance, eucl_distance_np
from environment import Environment, Social_environment
from visualizations import single_agent_animation, plot_single_agent_run
from agent_RL import Gina, Guido, SocialGuido

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
import tkinter as tk
import random
import pickle
import sys
import argparse
from tqdm import tqdm


import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical





def evaluate_parameters(env, device, duration, fs, starting_distances, starting_orientations, k, frequency, coupling_weights, n_oscillators):
    """

    wrapper to make mulitple runs of one set of parameters


    """
    # create agent with these parameters

    policy = Guido(device, fs, frequency, coupling_weights, k, n_oscillators).to(device)
    # make saving runs for each episode
    all_approach_scores = []   
    all_positions_x = []
    all_positions_y = [] 
    all_input_values = []
    all_phases = []
    all_phase_differences = []
    all_actions = []
    all_angles = []

   # do episodes for each parameter combination
    for starting_distance in starting_distances:
        for starting_orientation in starting_orientations:

            approach_score, position_x, position_y, input_values, phases, phase_differences, actions, angles = single_simulation(env, duration, fs, policy, n_oscillators, starting_distance, starting_orientation) 
            all_approach_scores.append(approach_score)
            # save the trajectories for this run
            all_positions_x.append(position_x)
            all_positions_y.append(position_y)

        all_input_values.append(input_values)
        all_phase_differences.append(phase_differences)
        all_phases.append(phases) 
        all_actions.append(actions)
        all_angles.append(angles)


    return all_approach_scores, all_positions_x, all_positions_y, all_input_values, all_phases, all_phase_differences, all_angles, all_actions

def single_simulation(env, duration, fs, policy, n_oscillators, starting_distance, starting_orientation):

    starting_position = np.array([0, -starting_distance])

    state = env.reset(starting_position, starting_orientation)

    # reset Guido
    if n_oscillators == 4:
        policy.reset(torch.tensor([0., .0, .0, 0.]))
    elif n_oscillators == 5:
        policy.reset(torch.tensor([0., 0., 0., 0., 0.]))

    # Complete the whole episode
    start_distance = env.distance
    input_values = np.zeros((2, fs * duration))

    phase_differences = np.zeros((n_oscillators, fs * duration))
    phases = np.zeros((n_oscillators, fs * duration))

    actions = []
    angles = []
    for t in range(duration * fs):
        #action, log_prob = policy.act(state)
        action, log_prob, output_angle = policy.act(state)
        #state, reward, done = env.step(action, 10)
        state, reward, done = env.step(output_angle.cpu().detach().numpy(), 10)
        if done:
                break 
        input_values[:, t] = policy.input.cpu().detach().numpy()
        phase_differences[:, t] = policy.phase_difference.cpu().detach().numpy() * env.fs
        phases[:, t] = policy.phases.cpu().detach().numpy()

        actions.append(env.orientation)
        angles.append(policy.output_angle.cpu().detach().numpy())

    end_distance = env.distance
    approach_score = 1 - (end_distance / start_distance)

    return approach_score, env.position_x, env.position_y, input_values, phases, phase_differences, actions, angles



def evaluate_parameters_social(env, device, fs, duration, starting_distances, starting_orientations, k, frequency, coupling_weights, social_sensitivity, n_oscillators, flavour, n_agents, plot):


     # create multiple agents with same parameters
    agents = []
    for i in range(n_agents):
        agent_id = i
        if flavour == "eco":
             policy = Guido(device, fs, frequency, coupling_weights, k, n_oscillators).to(device)
        elif flavour == "social":
            policy = SocialGuido(device, fs, frequency, coupling_weights, k, social_sensitivity, n_agents, i).to(device)
        else:
            print('flavour not recognized')
        agents.append(policy)

    approach_scores = []

    # do ten episodes for each parameter combination
    for starting_distance in starting_distances:
        for starting_orientation in starting_orientations:

            agent_starting_positions = [] 
            agent_starting_orientations = []
            orientations = np.random.uniform(starting_orientation - 0.5, starting_orientation + 0.5, n_agents)
            positions = np.random.uniform(starting_distance- 5, starting_distance + 5, n_agents)
            for a in range(n_agents):
                agent_starting_positions.append(np.array([0, positions[a]]))
                agent_starting_orientations.append(orientations[a])
            states = env.reset(agent_starting_positions, agent_starting_orientations, n_agents)
            start_distance = np.mean(env.distances)

            # reset the agent phases
            for a in range(n_agents):
                if flavour == 1:
                    agents[a].reset(torch.tensor([0., 0., 0., 0., 0.]))
                else:
                    agents[a].reset(torch.tensor([0., 0., 0., 0.,]))
                
                # make random starting position around the real position for the agents

           
            # Complete the whole episode
            for t in range(duration * fs):
                actions = []

                # make forward pass through each agent and collect all the agent
                for a in range(len(agents)):
                    if flavour == 1:
                        action, log_prob, output_angle = agents[a].act(states[a], np.array(env.agent_orientations), env.inter_agent_distances)
                    else:
                        action, log_prob, output_angle = agents[a].act(states[a])
                    actions.append(output_angle.cpu().detach().numpy())

                # let all agents perform their next action in the environment
                states, rewards, done = env.step(actions, 10)

                if done:
                    break 
            

                agent_scores_1 = []
                agent_scores_2 = []

                for a in range(n_agents):
                    start_distance_1 = eucl_distance_np(np.array([-100, 0]), agent_starting_positions[a])
                    end_distance_1 = eucl_distance_np(np.array([-100, 0]), env.agent_positions[a])
                    agent_scores_1.append(1 - (end_distance_1 / start_distance_1))

                    start_distance_2 = eucl_distance_np(np.array([100, 0]), agent_starting_positions[a])
                    end_distance_2 = eucl_distance_np(np.array([100, 0]), env.agent_positions[a])
                    agent_scores_2.append(1 - (end_distance_2 / start_distance_2))

                approach_score = np.min([np.mean(agent_scores_1), np.mean(agent_scores_2)])
                approach_scores.append(approach_score)

    return approach_scores


