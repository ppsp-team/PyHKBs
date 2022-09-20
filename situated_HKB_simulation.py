
import matplotlib.pyplot as plt
import torch
import numpy as np
from utils import symmetric_matrix, eucl_distance
from agent import Agent, Agent_single_oscillator
import time
from matplotlib import animation
import tkinter as tk
import cmath

fs = torch.as_tensor([20])
duration = torch.as_tensor([50]) # s
stimulus_position = torch.tensor([0., 0.]) # [m, m]
#stimulus_decay_rate = torch.as_tensor([0.5])
periodic_randomization = False

# instantiate an agent
agent_id = 1
stimulus_sensitivity = torch.as_tensor([6])
phase_coupling = torch.as_tensor([2.5])
anti_phase_coupling = torch.as_tensor([0.5])
initial_phases = torch.tensor([0.65, 0.]) # rad
frequencies = torch.tensor([1.0, 1.0]) # Hertz
movement_speed = torch.as_tensor([20]) # m/s
agent_position = torch.tensor([0., -50.]) # [m, m]
agent_orientation = torch.as_tensor([0.]) # rad


stimulus_position = [0, 0] # m, m
stimulus_decay_rate = 0.01 # in the environment
stimulus_scale = 1 # in the environment


agent_position = torch.tensor([0., -100.]) # [m, m]
agent_orientation = torch.as_tensor([0.]) # rad

agent = Agent_single_oscillator(fs, agent_id, stimulus_sensitivity, phase_coupling, anti_phase_coupling, initial_phases, frequencies, movement_speed, agent_position, agent_orientation)


fig, (ax1, ax2) = plt.subplots(1, 2)




# plot the environment with stimulus concentration
N = 1000
x = np.linspace(-150, 150, N)
y = np.linspace(-150, 150, N)
xx, yy = np.meshgrid(x, y)
xx, yy = np.meshgrid(x, y)
zz = np.sqrt(xx**2 + yy**2)   
zs = stimulus_scale * np.exp( - stimulus_decay_rate * zz)
environment_plot = ax1.contourf(x, y, zs)
ax1.axis('scaled')
#ax1.set_title('Approach score = ' + str(np.mean(approach_scores)) )
plt.colorbar(environment_plot, ax=ax1)

# generate time and initialize arrays to store trajectory
t = torch.linspace(start=0, end=float(duration), steps=int(duration * fs))
agent_position_x = np.zeros((len(t),))
agent_position_y = np.zeros((len(t),))

agent_phase_difference_1 = np.zeros((len(t),))
agent_phase_difference_2= np.zeros((len(t),))

angle_phases = np.zeros((len(t),))

agent_input = np.zeros((len(t),))

phases_1 = np.zeros((len(t),))
phases_2 = np.zeros((len(t),))

attractors_x = np.zeros((2, len(t)))
attractors_x[:] = np.nan
repellents_x = np.zeros((2, len(t)))
repellents_x[:] = np.nan


frequency_difference = 0
def HKBextended(phase, sensory_input): 
    "Extended HKB equation."
    return frequency_difference + sensory_input - phase_coupling * np.sin(phase) - 2 * anti_phase_coupling * np.sin(2 * phase)


phase_range = np.linspace(- 0.5*np.pi, 1.5 * np.pi, 360)

# perform simulation
for i in range(len(t)):
    # get the current eye positions of the agent
    left_eye_position, right_eye_position = agent.eye_positions()

    # calculate the stimulus intensity at the eye positions
    distance_left_eye =  eucl_distance(stimulus_position, left_eye_position)
# stimulus_gradient_left = stimulus_decay_rate * stimulus_scale * np.exp( - stimulus_decay_rate * distance_left_eye )
    stimulus_gradient_left = stimulus_scale * np.exp( - stimulus_decay_rate * distance_left_eye)

    distance_right_eye =  eucl_distance(stimulus_position, right_eye_position)
    stimulus_gradient_right = stimulus_scale * np.exp( - stimulus_decay_rate * distance_right_eye)


# stimulus_gradient_right = stimulus_decay_rate * stimulus_scale * np.exp( - stimulus_decay_rate * distance_left_eye )

    # get agent movement based on stimulus intensities
    agent_position, agent_orientation, phases, phase_differences = agent.next_timestep(i, stimulus_gradient_left, stimulus_gradient_right, periodic_randomization)

    agent_input[i] = agent.sensor_stimulation


    agent_phase_difference_1[i] = phase_differences[0] * fs
    agent_phase_difference_2[i] = phase_differences[1] * fs

    
    # save agent position for visualization
    agent_position_x[i] = agent_position[0]
    agent_position_y[i] = agent_position[1]

    phases_1[i] = phases[0]
    phases_2[i] = phases[1]

    
    angle_phases[i] = np.angle(np.exp(1j * (phases[0] - phases[1]))) #% 2 * torch.pi phases[0] - phases[1]

    # detect the position of possible attractors
    for p in range(1):#, len(phase_range - 1)):
        n_a = 0
        n_r = 0
        # indicate attractors in green
        if (HKBextended(phase_range[p], agent_input[i]) < 0) & (HKBextended(phase_range[p - 1], agent_input[i]) > 0):
            attractors_x[n_a, i] = p
            n_a +=1
        # indicate repellers in red
        elif (HKBextended(phase_range[p], agent_input[i]) > 0) & (HKBextended(phase_range[p - 1], agent_input[i]) < 0):
            repellents_x[n_r, i] = p
            n_r +=1
# visualize agent trajectory
ax1.plot(agent_position_x, agent_position_y, color = 'red')
ax2.plot(agent_phase_difference_1)
ax2.plot(agent_phase_difference_2)
ax2.plot(agent_input)
ax2.plot(angle_phases)

ax2.plot(agent_input)
ax2.plot(angle_phases)



ax1.set_xlim([-200,200])
ax1.set_ylim([-200,200])
plt.show()


fig, (ax1, ax4) = plt.subplots(1, 2)
fig.set_size_inches(12,5)
# plot the environment with stimulus concentration
N = 1000
x = np.linspace(-150, 150, N)
y = np.linspace(-150, 150, N)
xx, yy = np.meshgrid(x, y)
xx, yy = np.meshgrid(x, y)
zz = np.sqrt(xx**2 + yy**2)   
zs = stimulus_scale * np.exp( - stimulus_decay_rate * zz)
#zs = stimulus_scale * np.exp( - stimulus_decay_rate * zz)
environment_plot = ax1.contourf(x, y, zs)
ax1.axis('scaled')
#ax1.set_title('Approach score = ' + str(np.mean(approach_scores)) )
plt.colorbar(environment_plot, ax=ax1)


trajectory, = ax1.plot(0, 0, color = 'red')

#line_1, = ax2.plot(0, 0, linewidth = 1)
#line_2, = ax2.plot(0, 0, linewidth = 1)

#line_3, = ax3.plot(0, 0)
#line_4, = ax3.plot(0, 0)

#ax2.set_xlim([0, t[-1]])
#ax2.set_ylim([-1.1, 1.1])

#ax3.set_xlim([0, t[-1]])
#ax3.set_ylim([-1.1, 15])


phase_range = np.linspace(- 0.5*np.pi, 1.5 * np.pi, 360)




# Visualization of the phase space and find fixed points
ax4.plot(phase_range, np.zeros((len(phase_range))), color = 'black', linewidth = 1)

ax4.set_xlabel("phase " + r'$\phi$')
ax4.set_ylabel("phase change " + r'$\dot{\phi}$')
ax4.set_xlim([- 0.5*np.pi, 1.5 * np.pi])
ax4.set_ylim([-3, 10])

line_5, = ax4.plot(phase_range, HKBextended(phase_range, 0))#, color = 'lightblue')


#attractor_1 = ax4.scatter(1000, 1000, color='green', s = 100)
#attractor_2 = ax4.scatter(1000, 1000, color='green', s = 100)

#repellent_1 = ax4.scatter(1000, 1000, color='red', s = 100)
#repellent_2 = ax4.scatter(1000, 1000, color='red', s = 100)

phase_point = ax4.scatter(0, 0, s = 40, color = 'black')

def update_simulation(i):
    """
    Function that is called by FuncAnimation at every timestep
    updates the simulation by one timestep and updates the plots correspondingly
    """
    
    trajectory.set_xdata(agent_position_x[:i])
    trajectory.set_ydata(agent_position_y[:i])


   # line_1.set_xdata(t[:i])
    #line_1.set_ydata(np.sin(phases_1[:i]))

    #line_2.set_xdata(t[:i])
    #line_2.set_ydata(np.sin(phases_2[:i]))

    # blue = change of phases of motor 
   # line_3.set_xdata(t[:i])
    #line_3.set_ydata(agent_phase_difference_2[:i])

    # blue = input data to sensor
   # line_4.set_xdata(t[:i])
   # line_4.set_ydata(agent_input[:i])

    line_5.set_xdata(phase_range)
    line_5.set_ydata(HKBextended(phase_range, agent_input[i]))

    between_phases = np.abs(phases_1[i] - phases_2[i]) % 2*np.pi
    if between_phases > 1.5 * np.pi:
        between_phases -= 2 * np.pi
    phase_point.set_offsets((between_phases, HKBextended(between_phases, agent_input[i])))

  


    return trajectory, line_5, phase_point #line_1, line_2, line_5, line_3, line_4,  #, attractor_1, attractor_2, repellent_1, repellent_2

anim = animation.FuncAnimation(fig, update_simulation, frames = range(duration * fs), interval = 40,
        blit = True)

anim.save('SimpleHKBSimulation.gif')
plt.show()

plt.close()




def plot_attractors():
    if np.isnan(attractors_x[0,i]):
        attractor_1.set_offsets((1000, 1000))
    else:
        attractor_1.set_offsets((attractors_x[0,i], 0))

    if np.isnan(attractors_x[1,i]):
        attractor_2.set_offsets((1000, 1000))
    else:
        attractor_2.set_offsets((attractors_x[1,i], 0))

    if np.isnan(repellents_x[0,i]):
        repellent_1.set_offsets((1000, 1000))
    else:
        repellent_1.set_offsets((repellents_x[0,i], 0))

    if np.isnan(repellents_x[1,i]):
        repellent_2.set_offsets((1000, 1000))
    else:
        repellent_2.set_offsets((repellents_x[1,i], 0))