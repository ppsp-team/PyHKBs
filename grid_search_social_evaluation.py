import numpy as np
from utils import symmetric_matrix, eucl_distance, initiate_coupling_weights
from environment import Environment, Social_environment
from simulations import evaluate_parameters
from visualizations import single_agent_animation, plot_single_agent_run, plot_single_agent_multiple_trajectories, plot_multi_agent_run
from agent_RL import Gina, Guido, MultipleGuidos

import seaborn as sns
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation, gridspec
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker
import tkinter as tk
import random
import pickle
import ternary

import torch



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def calculate_movement_KOP(orientations):
   orientations_matrix = np.zeros((len(orientations), len(orientations[0])))
   for a in range(len(orientations)):
      orientations_matrix[a,:] = orientations[a]

   KOP_in_time = np.abs(np.mean(np.exp(1j * orientations_matrix), 0))
   KOP_std = np.std(KOP_in_time)
   KOP_mean = np.mean(KOP_in_time)
   return KOP_in_time, KOP_std, KOP_mean 


def calculate_wPLI(phase_1, phase_2):
   delta_phase = phase_1 - phase_2
   Im = np.imag(np.exp(1j*(delta_phase)))
   numer = np.abs(np.mean(np.abs(Im) * np.sign(Im)))
   denom = np.mean(np.abs(Im))

   if denom == 0:
      denom = 1
   return numer / denom


def calculate_inter_agent_PLV(phase_matrices, window_length, window_step, fs):
   # calculate windowed PLV
   window_start = 0
   window_end = int(window_start + window_length)
   window_step = int(window_step)
   simulation_length =int(np.size(phase_matrices[0], 1))
   plv_in_time = []
   plv_5_in_time = []
   wpli_in_time = []
   wpli_5_in_time = []
   interval_times = []
   _n_oscillators = np.size(phase_matrices[0], 0)
   n_agents = len(phase_matrices)
   agent_combinations = n_agents * (n_agents - 1) / 2

   # for the whole duration of the trial
   while (window_start + window_length) < simulation_length:
      interval_times.append((window_start + window_length/2)/fs)
      plv = 0
      counter = 0
      plv_5 = 0

      wpli = 0
      counter = 0
      wpli_5 = 0


      # loop through all the agent pairs
      for a_i in range(n_agents):
         for a_j in range(a_i+1, n_agents):
            # for all the oscillators
            for i in range(_n_oscillators):
               plv += np.abs(np.mean(np.exp(1j *(phase_matrices[a_i][i, window_start:window_end] - phase_matrices[a_j][i, window_start:window_end]))))
               wpli += calculate_wPLI(phase_matrices[a_i][i, window_start:window_end] , phase_matrices[a_j][i, window_start:window_end])
               if i == 4: # if you have a fifth oscillator
                  this_plv_5 = np.abs(np.mean(np.exp(1j *(phase_matrices[a_i][i, window_start:window_end] - phase_matrices[a_j][i, window_start:window_end]))))
                  this_wpli_5 = calculate_wPLI(phase_matrices[a_i][i, window_start:window_end] , phase_matrices[a_j][i, window_start:window_end])
                  plv_5 += this_plv_5
                  plv += this_plv_5
                  
                  wpli_5 += this_wpli_5
                  wpli += this_wpli_5
               counter += 1 
      window_start += window_step
      window_end += window_step
               

      plv_in_time.append(plv / counter)
      plv_5_in_time.append(plv_5 / (counter/_n_oscillators) )
      wpli_in_time.append(wpli / counter)
      wpli_5_in_time.append(wpli_5 / (counter/_n_oscillators) )

   mean_plv = np.mean(plv_in_time)
   mean_plv_5 = np.mean(plv_5_in_time)

   mean_wpli = np.mean(wpli_in_time)
   mean_wpli_5 = np.mean(wpli_5_in_time)


   return mean_plv, mean_wpli, mean_plv_5, mean_wpli_5, interval_times, plv_in_time, plv_5_in_time, wpli_in_time, wpli_5_in_time


def calculate_intra_agent_PLV(phase_matrices, window_length, window_step, fs):
   # INTRA
   window_start = 0
   window_end = int(window_start + window_length)
   window_step = int(window_step)
   simulation_length =int(np.size(phase_matrices[0], 1))
   plv_in_time = []
   wpli_in_time = []
   plv_5_in_time = []
   interval_times = []
   _n_oscillators = np.size(phase_matrices[0], 0)
   n_agents = len(phase_matrices)
   oscillator_combinations = _n_oscillators * (_n_oscillators - 1) / 2
  


   while (window_start + window_length) < simulation_length:
      interval_times.append((window_start + window_length/2)/fs)
      plv = 0
      wpli = 0
      counter = 0
      for a in range(len(phase_matrices)):
         phase_matrix = phase_matrices[a]
         for i in range(_n_oscillators):
            for j in range(i+1, _n_oscillators): # i+1 because dont want connection of oscillator with itself
               plv += np.abs(np.mean(np.exp(1j *(phase_matrix[i, window_start:window_end] - phase_matrix[j, window_start:window_end]))))
               wpli += calculate_wPLI(phase_matrices[a][i, window_start:window_end] , phase_matrices[a][j, window_start:window_end])
               counter += 1
      window_start += window_step
      window_end += window_step
      plv_in_time.append(plv / counter)
      wpli_in_time.append(wpli / counter)
   mean_plv = np.mean(plv_in_time)
   mean_wpli = np.mean(wpli_in_time)

   return mean_plv, mean_wpli, interval_times, plv_in_time, wpli_in_time





def visualize_grid_search(grid_results, dependent_var, x_axis, y_axis, other_parameters):
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
         

   plt.xlabel("stimulus ratio")
   plt.ylabel("starting angle between outer agents")
   plt.imshow(plotting_array, vmin = 0, vmax = 1)
   ax = plt.gca()
   xtick_labels = []
   for i in  np.linspace(0, 180, len(x_axis_values)):
      xtick_labels.append(str(i))
   ytick_labels =  []
   for i in  np.linspace(0, 1, len(y_axis_values)):
      ytick_labels.append(str(i))

   plt.xticks([0, 49], ["one gradient", "two equal gradients"])
   plt.yticks([0, 49], ["0 degrees ", "180 degrees"])
   plt.colorbar()
   plt.title('Dependence on environment')
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


def evaluate_social_agents(grid_results, simulation):

   for i in range(len(grid_results.index)):
      filename = r"results/PyHKB data/" + simulation + "/run_" + str(i) + ".pickle"
      # open the grid search results 
      with open(filename, "rb") as input_file:
         run = pickle.load(input_file)
      positions_x = run["x position"]
      positions_y = run["y position"]
      orientations = run["orientation"]
      agent_phases = run["phases"]
      
      print('run ' + str(i+1) + 'out of' +  str(len(grid_results.index)))  
      n_agents = len(positions_x)

      fs = 100
      window_length = int(fs)
      window_step = int(fs/2)# change to 10 for better calculations later
      intra_mean_plv, intra_mean_wpli, interval_times, intra_plv_in_time, intra_wpli_in_time = calculate_intra_agent_PLV(agent_phases, window_length, window_step, fs)
      inter_mean_plv, inter_mean_wpli, inter_mean_plv_5, inter_mean_wpli_5, interval_times, inter_plv_in_time, inter_plv_5_in_time, inter_wpli_in_time, inter_wpli_5_in_time = calculate_inter_agent_PLV(agent_phases, window_length, window_step, fs)
      KOP_in_time, KOP_std, KOP_mean = calculate_movement_KOP(orientations)



      grid_results.loc[i,"movement_stdKOP"] = KOP_std 
      grid_results.loc[i,"movement_meanKOP"] = KOP_mean

      grid_results.loc[i,"intra_PLV"] = intra_mean_plv
      grid_results.loc[i,"intra_wPLI"] = intra_mean_wpli

      grid_results.loc[i,"inter_PLV"] = inter_mean_plv
      grid_results.loc[i,"inter_wPLI"] = inter_mean_wpli

   return grid_results




def make_ternary_plot(grid_results, dependent_variable, ax = None):
   x_axis_values = np.sort(np.unique(grid_results['stimulus_sensitivity'].to_numpy()))
   y_axis_values = np.sort(np.unique(grid_results['social_sensitivity'].to_numpy()))

   # has to be first y and then x because is matrix indexing, not plotting function
   plotting_array = np.zeros((len(y_axis_values), len(x_axis_values)))

   points = []
   data = dict()
   for x in range(len(x_axis_values)):
      for y in range(len(y_axis_values)):
         plot_val = grid_results[grid_results['stimulus_sensitivity'] == x_axis_values[x]]
         plot_val = plot_val[plot_val['social_sensitivity'] == y_axis_values[y]]
         data_val = float(np.mean(plot_val[dependent_variable].to_numpy()))
         z = float(np.mean(plot_val['internal_connectivity'].to_numpy()))
         if np.isnan(z):
            z = 0
         if np.isnan(data_val):
            data_val = 0

         data[(x, y)] = data_val
         points.append((x, y, z))

   scale = 50
   fontsize = 12
   if not ax == None:
      figure, tax = ternary.figure(scale=scale, ax = ax)
     # figure.set_size_inches(12.5, 10)
   else:
      figure, tax = ternary.figure(scale=scale)


   if "PL" in dependent_variable:
      vmin = 0.95
      vmax = 1
   elif "performance" in dependent_variable:
      vmin = 0.
      vmax = 1.
   elif "meanKOP" in dependent_variable:
      vmin = 0
      vmax = 1
   elif "stdKOP" in dependent_variable:
      vmin = 0
      vmax = 0.35
   else: 
      vmin = None
      vmax = None

   tax.heatmap(data, vmin = vmin, vmax = vmax)
   tax.boundary()
   offset = 0.15
   tax.left_axis_label("Environment", fontsize=fontsize, offset = offset)
   tax.right_axis_label("Social", fontsize=fontsize, offset = offset)
   tax.bottom_axis_label("Internal ", fontsize=fontsize, offset = offset)
  # tax.set_title(dependent_variable, fontsize=20, weight='bold')


   fontsize = 20
   #tax.right_corner_label("Internal", fontsize=fontsize)
   #tax.top_corner_label("Social", fontsize=fontsize)
   #tax.left_corner_label("Environment", fontsize=fontsize)
   tax.ticks(axis='lbr',  multiple=5, linewidth=1, offset = 0.025)
   tax.clear_matplotlib_ticks()
   tax.get_axes().axis('off')
   #ternary.plt.savefig("C:/Users/Administrator/Documents/Google/PyHKB/" + simulation + "_" + dependent_variable + ".png")

   #ternary.plt.show()




fig, axs = plt.subplots(3, 5)
row = 0
for simulation in ["Social_GridSearchResults_4_eco_v3011", "Social_GridSearchResults_5_eco_v3011", "Social_GridSearchResults_5_social_v2811"]:



   # open the grid search results 
   with open(r"results/PyHKB data/" + simulation + "_evaluated.pickle", "rb") as input_file:
      grid_results = pickle.load(input_file)

   print(grid_results)

   dependent_variable = 'performance'#'performance'
   print(simulation)
   print(dependent_variable)
   make_ternary_plot(grid_results, dependent_variable, ax = axs[row, 0])
   dependent_variable = 'movement_meanKOP'#'performance'
   print(simulation)
   print(dependent_variable)
   make_ternary_plot(grid_results, dependent_variable, ax = axs[row, 1])

   dependent_variable = 'movement_stdKOP'#'performance'
   print(simulation)
   print(dependent_variable)
   make_ternary_plot(grid_results, dependent_variable, ax = axs[row, 2])


   dependent_variable = 'inter_PLV'#'performance'
   print(simulation)
   print(dependent_variable)
   make_ternary_plot(grid_results, dependent_variable, ax = axs[row, 3])


   dependent_variable = 'intra_PLV'#'performance'
   print(simulation)
   print(dependent_variable)
   make_ternary_plot(grid_results, dependent_variable, ax = axs[row, 4])

   row+=1
plt.show()

simulation = "Social_GridSearchResults_environment_4_eco_v3011"

with open(r"results/PyHKB data/" + simulation + ".pickle", "rb") as input_file:
   grid_results = pickle.load(input_file)


other_parameters = {"k": 2, "f_sens": 5, "f_motor": 5}
visualize_grid_search(grid_results, 'performance', 'stimulus_ratio', 'start_orientation', other_parameters)
#print(grid_results)


print(grid_results['stimulus_ratio'].unique())
print(grid_results['start_orientation'].unique())


performances = []
ratios = []
#for index in [index_0 , index_1, index_2, index_3, index_4]:
for stimulus_ratio in grid_results['stimulus_ratio'].unique():
   print(stimulus_ratio)
   for start_orientation in [grid_results['start_orientation'].unique()[39]]: #[8, 17, 25, 45]
      print(start_orientation)
      selected_rows = grid_results.loc[(grid_results['stimulus_ratio'] == stimulus_ratio) & (grid_results['start_orientation'] == start_orientation)]

      performances.append(selected_rows['performance'])
      ratios.append(np.round(stimulus_ratio, 2))
     # index= grid_results.iloc[(grid_results['stimulus_ratio'] == stimulus_ratio) & (grid_results['start_orientation'] == start_orientation)]
      print(list(np.where((grid_results['stimulus_ratio'] == stimulus_ratio) & (grid_results['start_orientation'] == start_orientation))))

      index = np.where((grid_results['stimulus_ratio'] == stimulus_ratio) & (grid_results['start_orientation'] == start_orientation))[0][0]
      
      #index = grid_results.index[grid_results['stimulus_ratio'] == stimulus_ratio & grid_results['start_orientation'] == start_orientation].tolist()[0]
      #print(index)
      filename = r"results/PyHKB data/" + simulation + "/run_" + str(index) + ".pickle"
      # open the grid search results 
      with open(filename, "rb") as input_file:
         run = pickle.load(input_file)
      positions_x = run["x position"]
      positions_y = run["y position"]
      orientations = run["orientation"]
      agent_phases = run["phases"]
   
      fig = plot_multi_agent_run(stimulus_ratio, 0.02, 1, positions_x, positions_y , 10)
      plt.title('ratio ' + str(stimulus_ratio) + ' | ' + 'orientation ' + str(start_orientation))
      plt.show()
    #  plt.savefig(r"results/PyHKB data/" + simulation + "/run_" + str(index) + ".png")
      plt.close()

plt.plot(ratios, performances, linewidth = 3)
plt.ylabel('Performance')
plt.xlabel('Stimulus ratio')
plt.show()

for simulation in ["Social_GridSearchResults_4_eco_v3011", "Social_GridSearchResults_5_eco_v3011", "Social_GridSearchResults_5_social_v2811"]:


   ######calculating part##################
   with open(r"results/PyHKB data/" + simulation + "_evaluated.pickle", "rb") as input_file:
      grid_results = pickle.load(input_file)

   plot_grid_results = grid_results[grid_results['stimulus_sensitivity'] == 30]

   fig, (ax1, ax2) = plt.subplots(2,1)#plt.figure(figsize=(10,6))
   sns.lineplot(data = plot_grid_results , x = 'social_sensitivity', y = 'performance', markers = True, dashes = False, ax = ax1)
   sns.lineplot(data = plot_grid_results , x = 'social_sensitivity', y = 'movement_meanKOP', markers = True, dashes = False, ax = ax1)
   sns.lineplot(data = plot_grid_results , x = 'social_sensitivity', y = 'movement_stdKOP', markers = True, dashes = False, ax = ax1)
   sns.lineplot(data = plot_grid_results , x = 'social_sensitivity', y = 'inter_wPLI', markers = True, dashes = False,ax = ax2, color = 'r')
   sns.lineplot(data = plot_grid_results , x = 'social_sensitivity', y = 'intra_wPLI', markers = True, dashes = False,ax = ax2, color = 'cyan')

   #ax1.legend([line1, line2, line3, line4, line5], ['performance', 'movement alignment', 'movement negotiation', 'inter-agent wPLI', 'intra_agent wPLI'])
   ax1.legend(labels = ['performance', 'movement alignment', 'movement negotiation'])
   ax2.legend(labels = ['inter-agent wPLI', 'intra-agent wPLI'])
   
   ax1.set_ylabel('Behavioral measure')
   ax2.set_ylabel('Brain measure')

   for i in [3, 8, 17, 25, 45]:
      ax1.axvline(x = i, color = 'black')
      ax2.axvline(x = i, color = 'black')

   plt.show()


index_1 = plot_grid_results.index[plot_grid_results['social_sensitivity'] == 8].tolist()[0]
index_2 = plot_grid_results.index[plot_grid_results['social_sensitivity'] == 17].tolist()[0]
index_3 = plot_grid_results.index[plot_grid_results['social_sensitivity'] == 25].tolist()[0]
index_4 = plot_grid_results.index[plot_grid_results['social_sensitivity'] == 45].tolist()[0]

simulation = "Social_GridSearchResults_4_eco_v3011"

#for index in [index_0 , index_1, index_2, index_3, index_4]:
for social_sensitivity in range(50): #[8, 17, 25, 45]
      print(social_sensitivity)
      index = plot_grid_results.index[plot_grid_results['social_sensitivity'] == social_sensitivity].tolist()[0]
      filename = r"results/PyHKB data/" + simulation + "/run_" + str(index) + ".pickle"
      # open the grid search results 
      with open(filename, "rb") as input_file:
         run = pickle.load(input_file)
      positions_x = run["x position"]
      positions_y = run["y position"]
      orientations = run["orientation"]
      agent_phases = run["phases"]
      plot_multi_agent_run(0.8, 0.02, 1, positions_x, positions_y , 10)








#grid_results.index = range(len(grid_results.index))
#grid_results["inter_PLV"] = ""


 
other_parameters = {"k": 2, "f_sens": 5, "f_motor": 5}
visualize_grid_search(grid_results, 'performance', 'stimulus_ratio', 'start_orientation', other_parameters)



grid_results = evaluate_social_agents(grid_results, simulation)

with open(r"results/PyHKB data/" + simulation + "_evaluated.pickle", "wb") as output_file: 
      pickle.dump(grid_results, output_file, protocol=pickle.HIGHEST_PROTOCOL)
############################################""

# open the grid search results 
with open(r"results/PyHKB data/" + simulation + "_evaluated.pickle", "rb") as input_file:
   grid_results = pickle.load(input_file)

      

#calculate_inter_PLV(grid_results)

#calculate_inter_PLV(grid_results)

#with open(r"results/PyHKB data/" + simulation + "_evaluated.pickle", "wb") as output_file: 
           # pickle.dump(grid_results, output_file, protocol=pickle.HIGHEST_PROTOCOL)
   


#with open(r"results/PyHKB data/" + simulation + "_evaluated.pickle", "rb") as input_file:
   #grid_results = pickle.load(input_file)




show_grid_search_results(grid_results, 10)

other_parameters = {"k": 2, "f_sens": 5, "f_motor": 5}




# make the data






visualize_grid_search(grid_results, 'performance', 'stimulus_sensitivity', 'internal_connectivity', other_parameters)
visualize_grid_search(grid_results, 'performance', 'social_sensitivity', 'internal_connectivity', other_parameters)
visualize_grid_search(grid_results, 'performance', 'social_sensitivity', 'stimulus_sensitivity', other_parameters)



# choose one of the runs
grid_results = average_grid_search(grid_results)





######calculating part##################
with open(r"results/PyHKB data/" + simulation + ".pickle", "rb") as input_file:
   grid_results = pickle.load(input_file)

grid_results = evaluate_social_agents(grid_results, simulation)

with open(r"results/PyHKB data/" + simulation + "_evaluated.pickle", "wb") as output_file: 
         pickle.dump(grid_results, output_file, protocol=pickle.HIGHEST_PROTOCOL)
############################################""



for i in range(len(grid_results.index)):  
   plot_grid_results = grid_results[grid_results['stimulus_sensitivity'] == 12]


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

