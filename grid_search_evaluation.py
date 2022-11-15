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


def average_grid_search(grid_results):
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
with open(r"results/GridSearchResults_4_random.pickle", "rb") as input_file:
   [grid_results, grid_runs] = pickle.load(input_file)

run = grid_runs[0]
x_position = run["x position"]
y_position = run["y position"]
plt.plot(x_position, y_position)
plt.show()
grid_results = average_grid_search(grid_results)
show_grid_search_results(grid_results, 10)


# choose one of the runs

print(run["end time"])

# for this function we have to adjust the data structure
# plot_single_agent_multiple_trajectories(all_positions_x, all_positions_y, stimulus_scale, stimulus_decay_rate, environment, stimulus_ratio)



x_position = run["x position"]
y_position = run["y position"]
phase_differences = run["phase differences"]
input_values = run["input values"]
angles = run["output angle"]
actions = run["orientation"]
phases = run["phases"]

plt.plot(x_position, y_position)
plt.show()


#plot_single_agent_run(f_sens, f_motor, coupling_weights, k, x_position, y_position, phase_differences, input_values, angles, actions, stimulus_scale, stimulus_ratio, stimulus_decay_rate)




#max_mean_agent, min_mean_agent, max_min_agent = find_agents(grid_results)
max_mean_agent = max_mean_agent.to_dict()
print(max_mean_agent)

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



