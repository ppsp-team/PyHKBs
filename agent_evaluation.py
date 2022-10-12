import numpy as np
from utils import symmetric_matrix, eucl_distance
from environment import Environment, Social_environment
from visualizations import single_agent_animation, plot_single_agent_run
from agent_RL import Gina, Guido, MultipleGuidos

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
import tkinter as tk
import random
import pickle





# visualize the grid search results
other_parameters = {"sensitivity": 5, "k": 5., "f_sens": 1., "f_motor": 1., "a_sens": 0.5, "a_ips": 1.5, "a_con": 0.4, "a_motor": 0.5, "scaling_factor": 0.1}
#visualize_grid_search(grid_results, "a_con", "a_ips", other_parameters)

# evaluate a specific combination of parameters
sensitivity = 5.
k = 5.
f_sens = 0.
f_motor = 0.
a_sens = 0.5
a_ips = 1.5
a_con = 0.4
a_motor = 0.5
n_episodes = 1

scaling = 0.1

starting_orientation = random.uniform(-np.pi, np.pi)
print(starting_orientation)
starting_position = np.array([0, -random.randrange(95, 105)])

env = Environment(fs, duration, stimulus_position, stimulus_decay_rate,
   stimulus_scale, sensitivity, starting_position, starting_orientation, movement_speed, agent_radius, agent_eye_angle, delta_orientation)

evaluate_parameters(env, n_episodes, k, f_sens, f_motor, a_sens*scaling, a_ips*scaling, a_con*scaling, a_motor*scaling, True)



def evaluate_parameters(env, starting_distances, starting_orientations, k, f_sens, f_motor, a_sens, a_ips, a_con, a_motor, plot):

   
   frequency = np.array([f_sens, f_motor])
   phase_coupling = np.array([a_sens, a_con, a_ips, a_motor])
   # create agent with these parameters
   policy = Guido(device, fs, frequency, phase_coupling, k).to(device)

   approach_scores = []   

   # do ten episodes for each parameter combination
   for starting_distance in starting_distances:
      for starting_orientation in starting_orientations:
         
         starting_position = np.array([0, -starting_distance])

         state = env.reset(starting_position, starting_orientation)

         # reset Guido
         policy.reset(torch.tensor([0., 0., 0., 0.]))

         # Complete the whole episode
         start_distance = env.distance
         input_values = np.zeros((2, fs * duration))
         phase_differences = np.zeros((4, fs * duration))
         phases = np.zeros((4, fs * duration))

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
         approach_scores.append(approach_score)


         if plot == True:
            fig = plot_single_agent_run(f_sens, f_motor, a_sens, a_motor, a_ips, a_con, k, env.position_x, env.position_y, phase_differences, input_values, angles, actions, stimulus_scale, stimulus_decay_rate)
            anim = single_agent_animation(env.position_x, env.position_y, phases, phase_differences, stimulus_scale, stimulus_decay_rate, duration, fs)
            #anim.save('GuidoSimulation.gif')

   return approach_scores



series : np.linspace(1,100,100)

def d(series,i,j):
    return abs(series[i]-series[j])
 
N=len(series)
eps= 100 # input diameter bound
dlist=[[] for i in range(N)]
n=0 #number of nearby pairs found
for i in range(N):
    for j in range(i+1,N):
        if d(series,i,j) < eps:
            n+=1
            print(n)
            for k in range(min(N-i,N-j)):
                dlist[k].append(np.log(d(series,i+k,j+k)))
f=open('lyapunov.txt','w')

for i in range(len(dlist)):
    if len(dlist[i]):
        print>>f, i, sum(dlist[i])/len(dlist[i])
f.close()




