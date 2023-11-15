import numpy as np
from utils import symmetric_matrix, eucl_distance, initiate_coupling_weights
from environment import Environment, Social_environment
from simulations import evaluate_parameters
from visualizations import single_agent_animation, plot_single_agent_run, plot_single_agent_multiple_trajectories
from agent_RL import Gina, Guido, MultipleGuidos

import seaborn as sns
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker
import tkinter as tk
import random
import pickle

import torch



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def calculate_KOP(phase_matrix):
   KOP_in_time = np.abs(np.mean(np.exp(1j * phase_matrix), 0))
   KOP_std = np.std(KOP_in_time)
   return KOP_in_time, KOP_std


def calculate_wPLI(phase_1, phase_2):
   delta_phase = phase_1 - phase_2
   Im = np.imag(np.exp(1j*(delta_phase)))
   numer = np.abs(np.mean(np.abs(Im) * np.sign(Im)))
   denom = np.mean(np.abs(Im))

   if denom == 0:
      denom = 1
   return numer / denom
   
def calculate_average_wPLI(phase_matrix, window_length, window_step):
   # calculate windowed PLV
   window_start = 0
   window_end = window_start + window_length
   simulation_length =int(np.size(phase_matrix, 1))
   plv_in_time = []
   interval_times = []
   _n_oscillators = np.size(phase_matrix, 0)
   oscillator_combinations = _n_oscillators * (_n_oscillators - 1) / 2


   while (window_start + window_length) < simulation_length:
      interval_times.append(window_start + window_length/2)
      plv = 0
      counter = 0
      for i in range(_n_oscillators):
         for j in range(i+1, _n_oscillators): # i+1 because dont want connection of oscillator with itself
            plv +=calculate_wPLI(phase_matrix[i, window_start:window_end], phase_matrix[j, window_start:window_end])
            window_start += window_step
            window_end += window_step
            counter += 1
      plv_in_time.append(plv / oscillator_combinations)
   mean_plv = np.mean(plv_in_time)

   return plv_in_time, interval_times, mean_plv





def calculate_average_PLV(phase_matrix, window_length, window_step):
   # calculate windowed PLV
   window_start = 0
   window_end = window_start + window_length
   simulation_length =int(np.size(phase_matrix, 1))
   plv_in_time = []
   interval_times = []
   _n_oscillators = np.size(phase_matrix, 0)
   oscillator_combinations = _n_oscillators * (_n_oscillators - 1) / 2


   while (window_start + window_length) < simulation_length:
      interval_times.append(window_start + window_length/2)
      plv = 0
      counter = 0
      for i in range(_n_oscillators):
         for j in range(i+1, _n_oscillators): # i+1 because dont want connection of oscillator with itself
            plv += np.abs(np.mean(np.exp(1j *(phase_matrix[i, window_start:window_end] - phase_matrix[j, window_start:window_end]))))
            window_start += window_step
            window_end += window_step
            counter += 1
      plv_in_time.append(plv / oscillator_combinations)
   mean_plv = np.mean(plv_in_time)

   return plv_in_time, interval_times, mean_plv



def visualize_grid_search(grid_results, dependent_var, x_axis, y_axis, other_parameters, ax = None, vmin = 0, vmax = 1):
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
   
   for key in other_parameters:
      if not ( (key == x_axis) or (key == y_axis) ):

         # make subselection of fixed parameters
         grid_results = grid_results[grid_results[key] == other_parameters[key]]

   # for the other parameters, make a numpy array to plot
   x_axis_values = np.sort(np.unique(grid_results[x_axis].to_numpy()))
   y_axis_values = np.sort(np.unique(grid_results[y_axis].to_numpy()))
   plotting_array = np.zeros((len(x_axis_values), len(y_axis_values)))

   # for the other parameters, make a numpy array to plot
   x_axis_values = np.sort(np.unique(grid_results[x_axis].to_numpy()))
   y_axis_values = np.sort(np.unique(grid_results[y_axis].to_numpy()))

   # has to be first y and then x because is matrix indexing, not plotting function
   plotting_array = np.zeros((len(y_axis_values), len(x_axis_values)))


   print(len(x_axis_values))
   print(len(y_axis_values))
   for x in range(len(x_axis_values)):
      for y in range(len(y_axis_values)):
         plot_val = grid_results[grid_results[x_axis] == x_axis_values[x]]
         plot_val = plot_val[plot_val[y_axis] == y_axis_values[y]]
         plotting_array[y, x] = float(np.mean(plot_val[dependent_var].to_numpy()))
         
   #plt.xticks(np.arange(0, len(x_axis_values), 1), x_axis_values)
   #plt.yticks(np.arange(0, len(y_axis_values), 1), y_axis_values)
   if not (ax == None):
      ax.imshow(plotting_array, vmin = vmin, vmax = vmax)
   else:
      plt.xlabel(x_axis)
      plt.ylabel(y_axis)
      plt.imshow(plotting_array, vmin = vmin, vmax = vmax)
      plt.colorbar()
      plt.show()


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
   grid_results = grid_results.groupby(["sensitivity", "k", "f_sens", "f_motor", "a_sens", "a_ips_left", "a_ips_right", "a_con_left", "a_con_right", "a_motor", "scaling_factor", "asymmetry_degree"]).agg({'performance': 'mean','stdKOP': 'mean', 'meanPLV': 'mean'})
   grid_results.columns = ['performance', 'stdKOP', 'meanPLV']
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


def evaluate_PLV_KOP(grid_results, simulation):
   grid_results.index = range(len(grid_results.index))
   grid_results["stdKOP"] = ""
   grid_results["meanPLV"] = ""

   for i in range(len(grid_results.index)):

      try:
         filename = r"results/PyHKB data/" + simulation + "/run_" + str(i) + ".pickle"
         with open(filename, "rb") as input_file:
            run = pickle.load(input_file)

         print('run ' + str(i+1) + 'out of' +  str(len(grid_results.index)))

         x_position = run["x position"]
         y_position = run["y position"]
         phase_differences = run["phase differences"]
         input_values = run["input values"]
         angles = run["output angle"]
         actions = run["orientation"]
         phases = run["phases"]

         fs = 100
         window_length = int(fs)
         window_step = int(fs/10)
         plv_in_time, interval_times, mean_plv = calculate_average_PLV(phases, window_length, window_step)
         wpli_in_time, interval_times, mean_wpli = calculate_average_wPLI(phases, window_length, window_step)

         KOP_in_time, KOP_std = calculate_KOP(phases)

         run["PLV"] = [interval_times, plv_in_time]
         run["KOP"] = KOP_in_time
   
         # save evaluated run

         grid_results.loc[i,"stdKOP"] = KOP_std 
         grid_results.loc[i,"meanPLV"] = mean_plv
         grid_results.loc[i,"meanwPLI"] = mean_wpli


      except:
         print('not found')
         grid_results.loc[i,"stdKOP"] = 0
         grid_results.loc[i,"meanPLV"] = 0
         
   return grid_results



simulation = "GridSearchResults_5_random_v1811"
#open the grid search results 


with open(r"results/PyHKB data/" + simulation + "_evaluated.pickle", "rb") as input_file:
   grid_results = pickle.load(input_file)



fig, (ax1, ax2) = plt.subplots(2,1, sharex = True)
palet = sns.color_palette("flare_r", as_cmap=True)
# make distribution plots


for i in range(len(grid_results.index)):
   if grid_results['performance'][i] < 0:
      grid_results['performance'][i] = 0
      

sns.scatterplot(ax = ax1, data = grid_results, x = 'scaling_factor', y = 'meanPLV', hue = 'performance', palette=palet, style = 'sensitivity', markers = ['s','o'], alpha = 0.3, s = 25)
ax1.xaxis.set_major_locator(ticker.LinearLocator(10))
ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax1.get_legend().remove()
# still have to fix that when rounding there the values are not correct anymore

sns.scatterplot(ax = ax2, data = grid_results, x = 'scaling_factor', y = 'stdKOP', hue = 'performance', palette=palet, style = 'sensitivity', markers = ['s','o'], alpha = 0.3, s = 25)
#ax2 = sns.stripplot(data = grid_results, x = 'scaling_factor', y = 'stdKOP', jitter = True, hue = 'sensitivity', palette="deep")
ax2.xaxis.set_major_locator(ticker.LinearLocator(10))
ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax2.set_xlabel('Internal coupling')
plt.subplots_adjust(hspace=.0)
plt.show()
    

# make distribution plots
palet = sns.color_palette("flare", as_cmap=True)
ax = sns.scatterplot(data = grid_results, x = 'meanPLV', y = 'performance', hue = 'scaling_factor', palette=palet, style = 'stimulus_ratio', markers = ['s','o'])

ax.set_xlim([0.99999, 1])
# still have to fix that when rounding there the values are not correct anymore
#plt.legend([],[], frameon= False)
plt.show()


# make distribution plots
ax = sns.scatterplot(data = grid_results, x = 'stdKOP', y = 'performance', hue = 'scaling_factor', palette=palet, style = 'stimulus_ratio', markers = ['s','o'])
ax.xaxis.set_major_locator(ticker.LinearLocator(10))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# still have to fix that when rounding there the values are not correct anymore
#plt.legend([],[], frameon= False)
plt.show()



other_parameters ={"asymmetry_degree": 0, "k": 2, "a_motor": 0, "stimulus_ratio": 0}
dependent = 'performance'
visualize_grid_search(grid_results, dependent, 'scaling_factor', 'sensitivity', other_parameters, vmin = 0, vmax = 1)
visualize_grid_search(grid_results, dependent, 'scaling_factor', 'sensitivity', other_parameters, vmin = 0.7, vmax = 1)
visualize_grid_search(grid_results, dependent, 'scaling_factor', 'sensitivity', other_parameters, vmin = 0, vmax = 0.3)



fig, axs = plt.subplots(6,2, sharex = 'col')

dependent = 'performance' # or 'performance' or 'meanPLV' or 'meanwPLI' or 'stdKOP'
other_parameters ={"asymmetry_degree": 0, "k": 2, "a_motor": 0, "stimulus_ratio": 0}
visualize_grid_search(grid_results, dependent, 'scaling_factor', 'sensitivity', other_parameters, ax = axs[0, 0], vmin = 0, vmax = 1)
dependent = 'meanPLV' # or 'performance' or 'meanPLV' or 'meanwPLI' or 'stdKOP'
other_parameters ={"asymmetry_degree": 0, "k": 2, "a_motor": 0, "stimulus_ratio": 0}
visualize_grid_search(grid_results, dependent, 'scaling_factor', 'sensitivity', other_parameters, ax = axs[1, 0], vmin = 0.7, vmax = 1)
dependent = 'stdKOP' # or 'performance' or 'meanPLV' or 'meanwPLI' or 'stdKOP'
other_parameters ={"asymmetry_degree": 0, "k": 2, "a_motor": 0, "stimulus_ratio": 0}
visualize_grid_search(grid_results, dependent, 'scaling_factor', 'sensitivity', other_parameters, ax = axs[2, 0], vmin = 0, vmax = 0.3)

dependent = 'performance' # or 'performance' or 'meanPLV' or 'meanwPLI' or 'stdKOP'
other_parameters ={"asymmetry_degree": 0, "k": 2, "a_motor": 1, "stimulus_ratio": 0}
visualize_grid_search(grid_results, dependent, 'scaling_factor', 'sensitivity', other_parameters, ax = axs[3, 0], vmin = 0, vmax = 1)
dependent = 'meanPLV' # or 'performance' or 'meanPLV' or 'meanwPLI' or 'stdKOP'
other_parameters ={"asymmetry_degree": 0, "k": 2, "a_motor": 1, "stimulus_ratio": 0}
visualize_grid_search(grid_results, dependent, 'scaling_factor', 'sensitivity', other_parameters, ax = axs[4, 0], vmin = 0.7, vmax = 1)
dependent = 'stdKOP' # or 'performance' or 'meanPLV' or 'meanwPLI' or 'stdKOP'
other_parameters ={"asymmetry_degree": 0, "k": 2, "a_motor": 1, "stimulus_ratio": 0}
visualize_grid_search(grid_results, dependent, 'scaling_factor', 'sensitivity', other_parameters, ax = axs[5, 0], vmin = 0, vmax = 0.3)

dependent = 'performance' # or 'performance' or 'meanPLV' or 'meanwPLI' or 'stdKOP'
other_parameters ={"asymmetry_degree": 0, "k": 2, "a_motor": 0, "stimulus_ratio": 0.95}
visualize_grid_search(grid_results, dependent, 'scaling_factor', 'sensitivity', other_parameters, ax = axs[0, 1], vmin = 0, vmax = 1)
dependent = 'meanPLV' # or 'performance' or 'meanPLV' or 'meanwPLI' or 'stdKOP'
other_parameters ={"asymmetry_degree": 0, "k": 2, "a_motor": 0, "stimulus_ratio": 0.95}
visualize_grid_search(grid_results, dependent, 'scaling_factor', 'sensitivity', other_parameters, ax = axs[1, 1], vmin = 0.7, vmax = 1)
dependent = 'stdKOP' # or 'performance' or 'meanPLV' or 'meanwPLI' or 'stdKOP'
other_parameters ={"asymmetry_degree": 0, "k": 2, "a_motor": 0, "stimulus_ratio": 0.95}
visualize_grid_search(grid_results, dependent, 'scaling_factor', 'sensitivity', other_parameters, ax = axs[2, 1], vmin = 0, vmax = 0.3)

dependent = 'performance' # or 'performance' or 'meanPLV' or 'meanwPLI' or 'stdKOP'
other_parameters ={"asymmetry_degree": 0, "k": 2, "a_motor": 1, "stimulus_ratio": 0.95}
visualize_grid_search(grid_results, dependent, 'scaling_factor', 'sensitivity', other_parameters, ax = axs[3, 1], vmin = 0, vmax = 1)
dependent = 'meanPLV' # or 'performance' or 'meanPLV' or 'meanwPLI' or 'stdKOP'
other_parameters ={"asymmetry_degree": 0, "k": 2, "a_motor": 1, "stimulus_ratio": 0.95}
visualize_grid_search(grid_results, dependent, 'scaling_factor', 'sensitivity', other_parameters, ax = axs[4, 1], vmin = 0.7, vmax = 1)
dependent = 'stdKOP' # or 'performance' or 'meanPLV' or 'meanwPLI' or 'stdKOP'
other_parameters ={"asymmetry_degree": 0, "k": 2, "a_motor": 1, "stimulus_ratio": 0.95}
visualize_grid_search(grid_results, dependent, 'scaling_factor', 'sensitivity', other_parameters, ax = axs[5, 1], vmin = 0, vmax = 0.3)
plt.show()




other_parameters ={"asymmetry_degree": 0, "k": 2, "a_motor": 1, 'sensitivity': 1.0}

for key in other_parameters:
   # make subselection of fixed parameters
   grid_results = grid_results[grid_results[key] == other_parameters[key]]



show_grid_search_results(grid_results, 100)












other_parameters ={"sensitivity": 5, "k": 2}

visualize_grid_search(grid_results, 'performance', 'scaling_factor', 'asymmetry_degree', other_parameters)
visualize_grid_search(grid_results, 'meanPLV', 'scaling_factor', 'asymmetry_degree', other_parameters)

show_grid_search_results(grid_results, 10)

other_parameters = {"sensitivity": 5, "k": 2, "f_sens": 5, "f_motor": 5, "a_sens": 0, "a_ips_left": 0, "a_ips_right": 0, "a_con_left": 1, "a_con_right": 1, "a_motor": 0}

grid_results = average_grid_search(grid_results)

visualize_grid_search(grid_results, 'stdKOP', 'scaling_factor', 'asymmetry_degree', other_parameters)
visualize_grid_search(grid_results, 'meanPLV', 'scaling_factor', 'asymmetry_degree', other_parameters)



######calculating part##################
with open(r"results/PyHKB data/" + simulation + ".pickle", "rb") as input_file:
   grid_results = pickle.load(input_file)

grid_results = evaluate_PLV_KOP(grid_results, simulation)

with open(r"results/PyHKB data/" + simulation + "_evaluated.pickle", "wb") as output_file: 
         pickle.dump(grid_results, output_file, protocol=pickle.HIGHEST_PROTOCOL)
############################################""

# choose one of the runs


# for this function we have to adjust the data structure
# plot_single_agent_multiple_trajectories(all_positions_x, all_positions_y, stimulus_scale, stimulus_decay_rate, environment, stimulus_ratio)




#plot_single_agent_run(f_sens, f_motor, coupling_weights, k, x_position, y_position, phase_differences, input_values, angles, actions, stimulus_scale, stimulus_ratio, stimulus_decay_rate)






run = grid_runs[0]
print(run["end time"])
x_position = run["x position"]
y_position = run["y position"]
phase_differences = run["phase differences"]
input_values = run["input values"]
angles = run["output angle"]
actions = run["orientation"]
phases = run["phases"]

plt.plot(x_position, y_position)
plt.show()

