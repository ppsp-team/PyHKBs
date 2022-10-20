import numpy as np
from utils import symmetric_matrix, eucl_distance, initiate_coupling_weights
from environment import Environment, Social_environment
from simulations import evaluate_parameters
from visualizations import single_agent_animation, plot_single_agent_run, plot_single_agent_multiple_trajectories
from agent_RL import Gina, Guido, MultipleGuidos

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
import tkinter as tk
import random
import pickle

import torch



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def show_grid_search_results(grid_results, rows):
   """
   look at parameters and find the ones that perform best
   """

   # find all values with a certain performance
   with pd.option_context('display.max_rows', rows,
                        'display.precision', 3,
                        ):
      print(grid_results)


def average_grid_serach(grid_results):
   """
   average the performance of all the runs with a certain set of parameters

   """
   grid_results = grid_results.groupby(["sensitivity", "k", "f_sens", "f_motor", "a_sens", "a_ips_left", "a_ips_right", "a_con_left", "a_con_right", "a_motor", "scaling_factor", "asymmetry_degree"]).agg({'performance': ['mean', 'min', 'max']})
   grid_results.columns = ['performance_mean', 'performance_min', 'performance_max']
   grid_results = grid_results.reset_index()
   return grid_results


def find_agents(grid_results):
   """
   finds the best agents
   the worst agent 
   and the max min agent
   
   """

   
   max_mean_agent = grid_results.iloc[grid_results["performance_mean"].argmax()]
   min_mean_agent = grid_results.iloc[grid_results["performance_mean"].argmin()]
   max_min_agent = grid_results.iloc[grid_results["performance_min"].argmax()]
   return  max_mean_agent, min_mean_agent, max_min_agent



# open the grid search results 
with open(r"results/GridSearchResults_single_stimulus_4_1810.pickle", "rb") as input_file:
   grid_results = pickle.load(input_file)

grid_results = average_grid_serach(grid_results)

#show_grid_search_results(grid_results, 10)

max_mean_agent, min_mean_agent, max_min_agent = find_agents(grid_results)
max_mean_agent = max_mean_agent.to_dict()
print(max_mean_agent)


# define variables for environment
fs = 50 # Hertz
duration = 20 # Seconds
stimulus_position = [0, 0] # m, m
stimulus_decay_rate = 0.01 # in the environment
stimulus_scale = 1 # in the environment
stimulus_sensitivity = 5 # of the agent
starting_position = [0, -100] 
starting_orientation = -0.25*np.pi
movement_speed = 10 #m/s
delta_orientation = 0.2*np.pi # rad/s turning speed
agent_radius = 5
agent_eye_angle = 45



# define variables for environment
fs = 50 # Hertz
duration = 100 # Seconds
stimulus_positions = [np.array([-100, 0]), np.array([100,0])] # m, m
stimulus_decay_rate = 0.02 # in the environment
stimulus_scale = 1 # in the environment
stimulus_sensitivity = 1 # of the agent
movement_speed = 10 #m/s
delta_orientation = 0.1*np.pi # rad/s turning speed # not used anymore here
stimulus_ratio = 1.
agent_radius = 2.5
agent_eye_angle = 0.5 * np.pi


intrinsic_frequencies = np.array([1., 1.])
n_oscillators = 4
starting_distances = [100]#np.linspace(95, 105, )
starting_orientations = np.linspace(-np.pi/2, np.pi/2, 5)

k = 2

environment = "double_stimulus"

a_sens = max_mean_agent["a_sens"]
a_ips_left = max_mean_agent["a_ips_left"]
a_ips_right= max_mean_agent["a_ips_right"]
a_con_left = max_mean_agent["a_con_left"]
a_con_right = max_mean_agent["a_con_right"]
a_motor = max_mean_agent["a_motor"]
scale = max_mean_agent["scaling_factor"]
stimulus_sensitivity = max_mean_agent["sensitivity"]
f_sens = max_mean_agent["f_sens"]
f_motor = max_mean_agent["f_motor"]
k = max_mean_agent["k"]


a_sens = 0.
a_ips_left = 0.1
a_ips_right= 0.1
a_con_left = 0.4
a_con_right = 0.4
a_motor = 0.5
scale = 1.5
stimulus_sensitivity = 5
f_sens = 0.2
f_motor = 0.2
k = 2

coupling_weights = scale * np.array([ a_sens, a_ips_left, a_ips_right, a_con_left, a_con_right, a_motor])
intrinsic_frequencies = np.array([f_sens, f_motor])






#for scale in np.linspace(0.1, 0.5, 5):
#coupling_weights = initiate_coupling_weights(scale, 0., False, 4)[0]
print(coupling_weights)



env = Environment(fs, duration, stimulus_positions, stimulus_ratio, stimulus_decay_rate,
   stimulus_scale, stimulus_sensitivity, starting_position, starting_orientation, movement_speed, agent_radius, agent_eye_angle, delta_orientation)

all_approach_scores, all_positions_x, all_positions_y, all_input_values, all_phases, all_phase_differences, all_angles, all_actions = evaluate_parameters(env, device, duration, fs, starting_distances, starting_orientations, k, intrinsic_frequencies, coupling_weights, n_oscillators)

plot_single_agent_multiple_trajectories(all_positions_x, all_positions_y, stimulus_scale, stimulus_decay_rate, environment, stimulus_ratio)


for x_position, y_position, phases, phase_differences, input_values, angles, actions in zip(all_positions_x, all_positions_y, all_phases, all_phase_differences, all_input_values, all_angles, all_actions):
   plot_single_agent_run(f_sens, f_motor, coupling_weights, k, x_position, y_position, phase_differences, input_values, angles, actions, stimulus_scale, stimulus_ratio, stimulus_decay_rate)
   single_agent_animation(x_position, y_position, phases, phase_differences, stimulus_scale, stimulus_decay_rate,  stimulus_ratio, duration, fs)


phase_matrix = all_phases[0]
for i in range(np.size(phase_matrix, 0)): 

   plt.plot(phase_matrix[i,:] % 2 * np.pi)

plv_in_time = np.exp(1j *(phase_matrix[0,:] - phase_matrix[1,:]))
plv = np.abs(np.mean(plv_in_time))
plt.plot(plv_in_time)
plt.show()
print()


# calculate average PLV between oscillators:





# get the phase locking value

print('loaded')







def visualize_grid_search(grid_results, x_axis, y_axis, other_parameters):
   """
   plots the results of the grid search on two specified dimension

   Arguments:
   -----------
   grid_results: pandas dataframe

   x_axis: string

   y_axis: string

   other_parameters: dictionary
      the values of the parameters that stay fixed

   
   """
   # extract requested values
   for key in other_parameters:
      if not ( (key == x_axis) or (key == y_axis) ):

         # make subselection of fixed parameters
         grid_results = grid_results[grid_results[key] == other_parameters[key]]

   # for the other parameters, make a numpy array to plot
   x_axis_values = np.sort(np.unique(grid_results[x_axis].to_numpy()))
   y_axis_values = np.sort(np.unique(grid_results[y_axis].to_numpy()))
   plotting_array = np.zeros((len(x_axis_values), len(y_axis_values)))

   
   for x in range(len(x_axis_values)):
      for y in range(len(y_axis_values)):
         plot_val = grid_results[grid_results[x_axis] == x_axis_values[x]]
         print(plot_val)
         plot_val = plot_val[grid_results[y_axis] == y_axis_values[y]]
         print(plot_val)
         plotting_array[x, y] = float(plot_val.performance.to_numpy())


   plt.xticks(np.arange(0, len(x_axis_values), 1), x_axis_values)
   plt.yticks(np.arange(0, len(y_axis_values), 1), y_axis_values)
   plt.xlabel(x_axis)
   plt.ylabel(y_axis)
   plt.imshow(plotting_array)
   plt.colorbar()
   plt.show()





