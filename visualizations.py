#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : visualizations.py
# description     : visualize multi agent or single agent runs
# author          : Nicolas Coucke
# date            : 2022-09-13
# version         : 1
# usage           : inside other scripts
# notes           : install the packages with "pip install -r requirements.txt"
# python_version  : 3.9.2
# ==============================================================================
from utils import symmetric_matrix, eucl_distance, eucl_distance_np
from environment import Environment, Social_environment, Social_stimulus_environment

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
import tkinter as tk
import random
import pickle


    
def plot_multi_agent_run(stimulus_ratio, stimulus_decay_rate, stimulus_scale, x_positions, y_positions, n_agents):
    fig, (ax)= plt.subplots(1,1)
    cmap = plt.cm.get_cmap("magma")
    # plot the trajectory of all agents in the environment
    for i in range(1,n_agents):
        x_prev = x_positions[i][0]
        y_prev = y_positions[i][1]
        t = 0
        for x, y in zip(x_positions[i], y_positions[i]):
            # later samples are more visible
            a = 0.1 + 0.5*t/len(x_positions[i])
            ax.plot([x_prev, x], [y_prev, y], alpha = a, color = cmap(i/n_agents))
            x_prev = x
            y_prev = y
            t+=1
        ax.set_xlim([-150, 150])
        ax.set_ylim([-150, 150])

    # plot the environment with stimulus concentration
    N = 1000
    x = np.linspace(-200, 200, N)
    y = np.linspace(-150, 150, N)
    xx, yy = np.meshgrid(x, y)
    xx, yy = np.meshgrid(x, y)
    zz_1 = np.sqrt((xx+100)**2 + yy**2)   
    zs_1 = stimulus_scale * np.exp( - stimulus_decay_rate * zz_1)
    #environment_plot = ax1.contourf(x, y, zs_1)

    zz_2 = np.sqrt((xx-100)**2 + yy**2)   
    zs_2 = stimulus_ratio*stimulus_scale * np.exp( - stimulus_decay_rate * zz_2)
    environment_plot = ax.contourf(x, y, zs_1 + zs_2)


    ax.axis('scaled')
    plt.colorbar(environment_plot, ax=ax)
    return fig


def draw_curve_horizontal(p1, p2):
    x = np.linspace(p1[0], p2[0], 100)

    height_difference = np.linspace(p1[1], p2[1], 100)

    y =  0.2*np.sin(np.pi *(x - p1[0])/2) + height_difference

    return x, y

def draw_curve_vertical(p1, p2):
    y = np.linspace(p1[1], p2[1], 100)

    height_difference = np.linspace(p1[0], p2[0], 100)

    x =  0.2*np.sin(np.pi *(y - p1[1])/2) + height_difference

    return x, y

def draw_anti_curve_horizontal(p1, p2):
    x = np.linspace(p1[0], p2[0], 100)

    height_difference = np.linspace(p1[1], p2[1], 100)

    y = - 0.2*np.sin(np.pi *(x - p1[0])/2) + height_difference

    return x, y

def draw_anti_curve_vertical(p1, p2):
    y = np.linspace(p1[1], p2[1], 100)

    height_difference = np.linspace(p1[0], p2[0], 100)

    x = - 0.2*np.sin(np.pi *(y - p1[1])/2) + height_difference

    return x, y

def plot_connection_strenghts(ax, f_sens, f_motor, coupling_values, k):
        
    # visualize the connection strengths
    # phase difference between oscillators
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    pos1 = [-1, 1]
    pos2 = [1, 1.001]
    pos3 = [-1.001, -1]
    pos4 = [1.001, -1.001]


    oscillator1 = ax.scatter(-1,1, s = f_sens * 200, color = 'blue') 
    oscillator2 = ax.scatter(1,1, s = f_sens * 200, color = 'lightblue')
    oscillator3 = ax.scatter(-1,-1, s = f_motor * 200, color = 'green')
    oscillator4 = ax.scatter(1,-1, s = f_motor * 200, color = 'lightgreen')

    a_sens = coupling_values[0]
    a_ips_left = coupling_values[1]
    a_ips_right = coupling_values[2]
    a_con_left = coupling_values[3]
    a_con_right = coupling_values[4]
    a_motor = coupling_values[5]

    # in-phase curves
    # sensor and motor connections
    x, y = draw_curve_horizontal(pos1, pos2)
    line_1_2, = ax.plot(x, y, color = 'blue', linewidth = a_sens)
    x, y = draw_anti_curve_horizontal(pos3, pos4)
    line_3_4, = ax.plot(x, y, color = 'blue', linewidth = a_motor)

    # ipsilateral connections
    x, y = draw_curve_vertical(pos1, pos3)
    line_1_3, = ax.plot(x, y, color = 'blue', linewidth = a_ips_left)
    x, y = draw_anti_curve_vertical(pos2, pos4)
    line_2_4, = ax.plot(x, y, color = 'blue', linewidth = a_ips_right)

    # contralateral connections
    x, y = draw_curve_horizontal(pos1, pos4)
    line_1_4, = ax.plot(x, y, color = 'blue', linewidth = a_con_left)
    x, y = draw_anti_curve_horizontal(pos2, pos3)
    line_2_3, = ax.plot(x, y, color = 'blue', linewidth = a_con_right)

    # anti-phase curves
    # sensor and motor connections
    x, y = draw_anti_curve_horizontal(pos1, pos2)
    line_1_2, = ax.plot(x, y, color = 'red', linewidth = a_sens / k)
    x, y = draw_curve_horizontal(pos3, pos4)
    line_3_4, = ax.plot(x, y, color = 'red', linewidth = a_motor / k)

    # ipsilateral connections
    x, y = draw_anti_curve_vertical(pos1, pos3)
    line_1_3, = ax.plot(x, y, color = 'red', linewidth = a_ips_left / k)
    x, y = draw_curve_vertical(pos2, pos4)
    line_2_4, = ax.plot(x, y, color = 'red', linewidth = a_ips_right / k)

    # contralateral connections
    x, y = draw_anti_curve_horizontal(pos1, pos4)
    line_1_4, = ax.plot(x, y, color = 'red', linewidth = a_con_left / k)
    x, y = draw_curve_horizontal(pos2, pos3)
    line_2_3, = ax.plot(x, y, color = 'red', linewidth = a_con_right / k)

    ax.axis('scaled')


def plot_single_agent_run(f_sens, f_motor, coupling_values, k, x_position, y_position, phase_differences, input_values, angles, actions, stimulus_scale, stimulus_ratio, stimulus_decay_rate):

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
    # plot the trajectory of the agant in the environment
    i = 0
    x_prev = x_position[0]
    y_prev = y_position[0]
    for x, y in zip(x_position, y_position):
        # later samples are more visible
        a = 0.1 + 0.5*i/len(x_position)
        ax1.plot([x_prev, x], [y_prev, y], alpha = a, color = 'red')
        x_prev = x
        y_prev = y
        i+=1
    ax1.set_xlim([-150, 150])
    ax1.set_ylim([-150, 150])


    plot_connection_strenghts(ax2, f_sens, f_motor, coupling_values, k)
    ax2.axis('scaled')

    ax3.plot(phase_differences[0, 2:len(angles)], color = 'blue')
    ax3.plot(phase_differences[1, 2:len(angles)], color = 'lightblue')

    ax3.plot(phase_differences[2, 2:len(angles)], color = 'green')
    ax3.plot(phase_differences[3, 2:len(angles)], color = 'lightgreen')

    ax3.plot(input_values[0, 2:len(angles)], color = 'blue', linewidth = 0.8, linestyle = '--')
    ax3.plot(input_values[1, 2:len(angles)], color = 'lightblue', linewidth = 0.8, linestyle = '--')


    # plot the output angle and actions
    ax4.plot(angles)
    ax4.plot(actions)


    # plot the environment with stimulus concentration
    N = 1000
    x = np.linspace(-200, 200, N)
    y = np.linspace(-150, 150, N)
    xx, yy = np.meshgrid(x, y)
    xx, yy = np.meshgrid(x, y)
    zz_1 = np.sqrt((xx+100)**2 + yy**2)   
    zs_1 = stimulus_scale * np.exp( - stimulus_decay_rate * zz_1)
    #environment_plot = ax1.contourf(x, y, zs_1)

    zz_2 = np.sqrt((xx-100)**2 + yy**2)   
    zs_2 = stimulus_ratio*stimulus_scale * np.exp( - stimulus_decay_rate * zz_2)
    environment_plot = ax1.contourf(x, y, zs_1 + zs_2)

    plt.colorbar(environment_plot, ax=ax1)
    plt.show()

    return fig




def plot_single_agent_multiple_trajectories(x_positions, y_positions, stimulus_scale, stimulus_decay_rate, environment, stimulus_ratio):
    fig, (ax1)= plt.subplots(1,1)
    cmap = plt.cm.get_cmap("magma")
    # plot the trajectory of all agents in the environment
    for i in range(len(x_positions)):
        x_prev = x_positions[i][0]
        y_prev = y_positions[i][0]
        t = 0
        for x, y in zip(x_positions[i], y_positions[i]):
            # later samples are more visible
            a = 0.1 + 0.5*t/len(x_positions[i])
            ax1.plot([x_prev, x], [y_prev, y], alpha = a, color = cmap(i/len(x_positions)))
            x_prev = x
            y_prev = y
            t+=1

    if environment == "single_stimulus":
        ax1.set_xlim([-500, 500])
        ax1.set_ylim([-550, 550])
    else:
        ax1.set_xlim([-200, 200])
        ax1.set_ylim([-150, 150])

    # plot the environment with stimulus concentration
    N = 1000
    x = np.linspace(-200, 200, N)
    y = np.linspace(-150, 150, N)
    xx, yy = np.meshgrid(x, y)
    xx, yy = np.meshgrid(x, y)
    zz_1 = np.sqrt((xx+100)**2 + yy**2)   
    zs_1 = stimulus_scale * np.exp( - stimulus_decay_rate * zz_1)
    #environment_plot = ax1.contourf(x, y, zs_1)

    zz_2 = np.sqrt((xx-100)**2 + yy**2)   
    zs_2 = stimulus_ratio*stimulus_scale * np.exp( - stimulus_decay_rate * zz_2)
    environment_plot = plt.contourf(x, y, zs_1 + zs_2)
    plt.axis('scaled')
    plt.colorbar(environment_plot)
    plt.show()
    return fig





def single_agent_animation(x_position, y_position, phases, phase_differences, stimulus_scale, stimulus_decay_rate, stimulus_ratio, duration, fs):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(12,5)
    # plot the environment with stimulus concentration
    N = 1000
    x = np.linspace(-200, 200, N)
    y = np.linspace(-150, 150, N)
    xx, yy = np.meshgrid(x, y)
    xx, yy = np.meshgrid(x, y)
    zz_1 = np.sqrt((xx+100)**2 + yy**2)   
    zs_1 = stimulus_scale * np.exp( - stimulus_decay_rate * zz_1)
    #environment_plot = ax1.contourf(x, y, zs_1)

    zz_2 = np.sqrt((xx-100)**2 + yy**2)   
    zs_2 = stimulus_ratio*stimulus_scale * np.exp( - stimulus_decay_rate * zz_2)
    environment_plot = ax1.contourf(x, y, zs_1 + zs_2)
    ax1.axis('scaled')
    #ax1.set_title('Approach score = ' + str(np.mean(approach_scores)) )
    plt.colorbar(environment_plot, ax=ax1)


    trajectory, = ax1.plot(0, 0, color = 'red')

    # phase difference between oscillators
    ax2.set_xlim([-1.5, 1.5])
    ax2.set_ylim([-1.5, 1.5])


    oscillator1 = ax2.scatter(-1,1, s = 40, animated=True) 
    oscillator2 = ax2.scatter(1,1, s = 40, animated=True)
    oscillator3 = ax2.scatter(-1,-1, s = 40, animated=True)
    oscillator4 = ax2.scatter(1,-1, s = 40, animated=True)
    oscillators = [oscillator1, oscillator2, oscillator3, oscillator4]
    line_1_2, = ax2.plot([-1, 1], [1, 1], color = 'grey')
    line_1_3, = ax2.plot([-1, -1], [1, -1], color = 'grey')
    line_1_4, = ax2.plot([-1, 1], [1, -1], color = 'grey')
    line_2_3, = ax2.plot([1, -1], [1, -1], color = 'grey')
    line_2_4, = ax2.plot([1, 1], [1, -1], color = 'grey')
    line_3_4, = ax2.plot([-1, 1], [-1, -1], color = 'grey')
    lines = [line_1_2, line_1_3, line_1_4, line_2_3, line_2_4, line_3_4]
    line_oscillators = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]

    patch1 = plt.Circle((-1, 1), 0.4, fc='orange')
    patch2 = plt.Circle((1, 1), 0.4, fc='orange')
    patch3 = plt.Circle((-1, -1), 0.4, fc='blue')
    patch4 = plt.Circle((1, -1), 0.4, fc='blue')
    patches = [patch1, patch2, patch3, patch4]
    ax2.add_patch(patch1)
    ax2.add_patch(patch2)
    ax2.add_patch(patch3)
    ax2.add_patch(patch4)

    ax2.set_title('Phase difference between oscillators')

    def update_simulation(i):
        """
        Function that is called by FuncAnimation at every timestep
        updates the simulation by one timestep and updates the plots correspondingly
        """
        
        trajectory.set_xdata(x_position[:i])
        trajectory.set_ydata(y_position[:i])

            # set the thickness of the edges to the phase differences between oscillators
        for L in range(6):
            between_oscillator = np.abs(phases[line_oscillators[L][0]-1, i] - phases[line_oscillators[L][1]-1, i]) % 2 * np.pi
            lines[L].set_linewidth(between_oscillator )

        for O in range(4):
            patches[O].radius = 0.1 + 0.02 * np.abs(phase_differences[O,i]) 

        return trajectory, line_1_2, line_1_3, line_1_4, line_2_3, line_2_4, line_3_4, patch1, patch2, patch3, patch4


    anim = animation.FuncAnimation(fig, update_simulation, frames = range(duration * fs), interval = 40,
            blit = True)

    plt.show()
    return anim




def multi_agent_animation(fig, ax, stimulus_ratio, stimulus_decay_rate, stimulus_scale, x_positions, y_positions, n_agents, duration, fs):
    cmap = plt.cm.get_cmap("magma")
    # make simulation
    # plot trajectories and angles of all players


    N = 1000
    x = np.linspace(-200, 200, N)
    y = np.linspace(-150, 150, N)
    xx, yy = np.meshgrid(x, y)
    xx, yy = np.meshgrid(x, y)
    zz_1 = np.sqrt((xx+100)**2 + yy**2)   
    zs_1 = stimulus_scale * np.exp( - stimulus_decay_rate * zz_1)
    #environment_plot = ax1.contourf(x, y, zs_1)

    zz_2 = np.sqrt((xx-100)**2 + yy**2)   
    zs_2 = stimulus_ratio * stimulus_scale * np.exp( - stimulus_decay_rate * zz_2)
    environment_plot = ax.contourf(x, y, zs_1 + zs_2)

    ax.axis('scaled')

    plt.colorbar(environment_plot, ax=ax)
    
    # ax2.set_xlim([0, duration])
    # ax2.set_ylim([-np.pi, np.pi])
    # ax2.set_title('agent orientations')

    trajectories = []
    orientations = []
    for a in range(n_agents):
        # add trajectory
        line_object, = ax.plot(0,0, color = cmap(a/n_agents))
        trajectories.append(line_object)

    t = np.linspace(0, duration, duration * fs)
    def update_simulation(i):
        """
        Function that is called by FuncAnimation at every timestep
        updates the simulation by one timestep and updates the plots correspondingly
        """
        for a in range(n_agents):
            trajectories[a].set_xdata(x_positions[a][:i])
            trajectories[a].set_ydata(y_positions[a][:i])

        return trajectories #+ orientations # use + to concatenate lists
    anim = animation.FuncAnimation(fig, update_simulation, frames = range(duration * fs), interval = 40,
            blit = True)

    return anim
   