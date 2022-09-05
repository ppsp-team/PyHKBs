#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : agent.py
# description     : Agent class with input and output
""""
This is an asocial agent that senses changes in stimulus intensity
Changes in stimulus intensity change the phase of the sensory oscillators
oscillator 1: left eye
oscillator 2: right eye
oscillator 3: left motor
oscillator 4: right motor
The agent travels at constant speed. The orientation of the agent changes according 
to the phase difference between the two motor oscillators

Structure:
-----------
next_timestep calls
    runge_kutta_HKB calls
        oscillator_system cals
            single_oscillator


"""
# author          : Nicolas Coucke
# date            : 2022-07-13
# version         : 1
# usage           : python agent.py
# notes           : install the packages with "pip install -r requirements.txt"
# python_version  : 3.9.2
# ==============================================================================

import matplotlib.pyplot as plt
import torch
import numpy as np
import cmath

# define sampling frequency
fs = torch.as_tensor([25])  # Hertz


class Agent:
    def __init__(self, agent_id, stimulus_sensitivity, phase_coupling_matrix, anti_phase_coupling_matrix, initial_phases, frequencies, movement_speed, initial_position, initial_orientation):

        # define agent variables that stay constant during the simulation
        self.id = agent_id # important later in social phase
        self.phase_coupling_matrix = phase_coupling_matrix # 4 by 4 matrix
        self.anti_phase_coupling_matrix = anti_phase_coupling_matrix # 4 by 4 matrix
        self.initial_phases = initial_phases # initial phases of the 4 oscillators
        self.frequencies = frequencies # intrinsic frequencies of the 4 oscillators
        self.stimulus_sensitivity = stimulus_sensitivity # of the agent to the change in stimulus intensity
        self.movement_speed = movement_speed # of the agent in the environment
        self.number_of_oscillators = 4
        self.agent_radius = torch.as_tensor([2.]) # radius of the agent's body
        self.agent_eye_angle = torch.pi/4 # 45 degree angle between the eyes of the agent

        # initialize agent variables that change during the simulation
        self.phases = initial_phases # phases of the 4 oscillators
        self.stimulus_intensity_left = torch.as_tensor([0.])
        self.stimulus_intensity_right = torch.as_tensor([0.])
        self.stimulus_change_left = torch.as_tensor([0.]) # change in stimulus intensity
        self.stimulus_change_right = torch.as_tensor([0.])
        self.position = initial_position # of agent in the environment
        self.orientation = initial_orientation # wrt world coordinate system



    def single_oscillator(self, oscillator_number, phases):
        """"
        The phase of a oscillator i is modified by being connected to the other oscillator j by means of the HKB equations
        If the oscillator is part of the sensory system, then its phase is also influenced by the change in stimulus intensity

        Arguments
        ----------

        oscillator number of oscillator i: int
            0, 1, 2 or 3 (true oscillator number -1)

        phases: torch.tensor of length 4
            the phase of each oscillator at the previous timestep

        Returns
        --------

        phase_difference: torch.tensor of length 1
            phase change of oscillator i

        """

        # extract phase of oscillator i
        oscillator_phase = phases[oscillator_number]

        # phase change of oscillator i due to intrinsic frequency
        phase_difference = torch.as_tensor([2 * torch.pi * self.frequencies[oscillator_number]])

        # include sensory input for the eyes
        if oscillator_number == 0: # left eye
            phase_difference += self.stimulus_sensitivity * self.stimulus_change_left
        elif oscillator_number == 1: # right eye
            phase_difference += self.stimulus_sensitivity * self.stimulus_change_right

        for other_oscillator_number in range(4): # loop through all oscillators j

            if other_oscillator_number != oscillator_number: # no self-coupling

                # get the phase and coupling variables for oscillator j
                phase_coupling = self.phase_coupling_matrix[oscillator_number,
                    other_oscillator_number]
                anti_phase_coupling = self.anti_phase_coupling_matrix[oscillator_number, 
                    other_oscillator_number]
                other_oscillator_phase = phases[other_oscillator_number]
#
                # phase change of oscillator i due to oscillator j
                between_oscillator_phase = (oscillator_phase - other_oscillator_phase) #% 2 * torch.pi
                phase_difference += torch.as_tensor([- phase_coupling * torch.sin(between_oscillator_phase) - 
                anti_phase_coupling * torch.sin(2 * (between_oscillator_phase))])

        return phase_difference


    def oscillator_system(self, phases):
        """
        Updates phases of the system of 4 mutually influencing oscillators 

        Arguments
        ----------

        phases: torch.tensor of length 4
            phases of the oscillators at the previous timestep


        Returns
        -----------

        phase_differences: torch.tensor of length 4
            phase differences for the 4 oscillators
            i.e. phases(t+1) - phases(t)

        """
        phase_differences = torch.zeros(4)

        # calculate updated phase for each oscillator individually
        for oscillator_number in range(self.number_of_oscillators):
            phase_differences[oscillator_number] = self.single_oscillator(oscillator_number,phases)

        return phase_differences


    def runge_kutta_HKB(self, phases):
        """
        Determine the phases of the oscillators at the current timestep
        using the Runge Kutta method of order 4

        Arguments
        ----------

        phases: torch.tensor of length 4
            phases of the oscillators at the previous timestep


         Returns
        -----------

        new_phases: torch.tensor of length 4
            the phases of the oscillators at the current timestep
          

        """
        k1 = self.oscillator_system(phases) * (1/fs)
        k2 = self.oscillator_system(phases + 0.5 * k1) * (1/fs)
        k3 = self.oscillator_system(phases + 0.5 * k2)  * (1/fs)
        k4 = self.oscillator_system(phases + k3) * (1/fs)
        phase_differences = (1/6) * (k1 + 2 * k2 + 2 * k3 + k4)
        new_phases = phases + phase_differences 
       # new_phases = new_phases % 2 * torch.pi # introduce periodicity

        return new_phases, phase_differences
        

    def next_timestep(self, time, stimulus_intensity_left, stimulus_intensity_right, periodic_randomization):
        """"
        Receive input from the environment
        Update the internal states of the agent
        Output the agent's new position and orientation to the environment
        
        Arguments: 
        -----------
            stimulus_intensity_left: float
                stimulus intensity at position of left eye
                stimulus intensity at position of right eye

        Returns:
        ----------
            position (x, y): torch.tensor
                avatar position in world space

            orientation: torch.tensor
                the orientation of the avatar in world space
                proportional to phase difference between motor oscillators


        """
        if periodic_randomization:
            # randomize phases every 20 timesteps (c.f. Aguilera et al., 2013)
            self.phases = self.randomize_phases(time)

        # calculate change in stimulus intensity for each eye
        self.stimulus_change_left = stimulus_intensity_left - self.stimulus_intensity_left
        self.stimulus_change_right = stimulus_intensity_right - self.stimulus_intensity_right

        # save intensity for next iteration
        self.stimulus_intensity_left = stimulus_intensity_left
        self.stimulus_intensity_right = stimulus_intensity_right

        # determine the new state of the agent
        self.phases, phase_differences  = self.runge_kutta_HKB(self.phases)

        # change in orientation is proportional to phase difference between motor units
        motor_phase_difference = self.phases[2] - self.phases[3] #% 2 * torch.pi 
        #motor_phase_difference = np.exp(self.phases[2] - self.phases[3]) #% 2 * torch.pi 

        motor_phase_difference = np.angle(np.exp(1j * (self.phases[2].detach().numpy() - self.phases[3].detach().numpy()))) #% 2 * torch.pi 
        self.orientation = self.orientation + torch.as_tensor(motor_phase_difference)

        # calculate next position according to movement speed and new orientation
        self.position = self.position + torch.tensor([torch.sin(self.orientation)
         * self.movement_speed * (1/fs), torch.cos(self.orientation) * self.movement_speed * (1/fs)])

        return self.position, self.orientation, np.array(self.phases), np.array(phase_differences)


    def eye_positions(self):
        """"
        Calculate position of the agent's eyes in world space
        based on the orientation and position
        
        Arguments: 
        -----------
            None; uses variables stored in the class

        Returns:
        ----------
            left_eye_position (x, y): torch.tensor of length 2
                position of the left eye in world space

            right_eye_position (x, y): torch.tensor of length 2
                position of the right eye in world space

        """
        left_eye_position = torch.zeros(2)
        right_eye_position = torch.zeros(2)

        left_eye_position[0] = self.position[0] + torch.sin(self.orientation - self.agent_eye_angle/2 ) * self.agent_radius
        left_eye_position[1] = self.position[1] + torch.cos(self.orientation - self.agent_eye_angle/2 ) * self.agent_radius

        right_eye_position[0] = self.position[0] + torch.sin(self.orientation + self.agent_eye_angle/2 ) * self.agent_radius
        right_eye_position[1] = self.position[1] + torch.cos(self.orientation + self.agent_eye_angle/2 ) * self.agent_radius

        return left_eye_position, right_eye_position 


    def randomize_phases(self, time):
        """
        Takes in the current phases of each oscillator and
        assigns random values to them from a uniform distribution
        between 0 and 2pi

        """
        if time % 20 == 0:
            self.phases = 2 * torch.pi * torch.rand(len(self.phases))

        return self.phases





class Agent_single_oscillator:
    def __init__(self, fs, agent_id, stimulus_sensitivity, phase_coupling, anti_phase_coupling, initial_phases, frequencies, movement_speed, initial_position, initial_orientation):

        # define agent variables that stay constant during the simulation
        self.fs = fs
        self.id = agent_id # important later in social phase
        self.phase_coupling = phase_coupling # float
        self.anti_phase_coupling= anti_phase_coupling # float
        self.initial_phases = initial_phases # initial phase
        self.frequencies = frequencies # intrinsic frequencies of the 4 oscillators
        self.stimulus_sensitivity = stimulus_sensitivity # of the agent to the change in stimulus intensity
        self.movement_speed = movement_speed # of the agent in the environment
        self.number_of_oscillators = 2
        self.agent_radius = torch.as_tensor([2.]) # radius of the agent's body
        self.agent_eye_angle = torch.pi/4 # 45 degree angle between the eyes of the agent

        # initialize agent variables that change during the simulation
        self.phases = initial_phases # phases of the 4 oscillators
        self.stimulus_intensity_left = torch.as_tensor([0.])
        self.stimulus_intensity_right = torch.as_tensor([0.])
        self.stimulus_change_left = torch.as_tensor([0.]) # change in stimulus intensity
        self.stimulus_change_right = torch.as_tensor([0.])
        self.position = initial_position # of agent in the environment
        self.orientation = initial_orientation # wrt world coordinate system



    def single_oscillator(self, oscillator_number, phases):
        """"
        The phase of a oscillator i is modified by being connected to the other oscillator j by means of the HKB equations
        If the oscillator is part of the sensory system, then its phase is also influenced by the change in stimulus intensity

        Arguments
        ----------

        oscillator number of oscillator i: int
            0, 1, 2 or 3 (true oscillator number -1)

        phases: torch.tensor of length 4
            the phase of each oscillator at the previous timestep

        Returns
        --------

        phase_difference: torch.tensor of length 1
            phase change of oscillator i

        """

        # extract phase of oscillator i
        oscillator_phase = phases[oscillator_number]

        # phase change of oscillator i due to intrinsic frequency
        phase_difference = torch.as_tensor([2 * torch.pi * self.frequencies[oscillator_number]])

        # include sensory input for the eyes
        if oscillator_number == 0: # sensor
            phase_difference += self.stimulus_sensitivity * 0.5 *(self.stimulus_change_left + self.stimulus_change_right)
      
        for other_oscillator_number in range(2): # loop through all oscillators j

            if other_oscillator_number != oscillator_number: # no self-coupling

                # get the phase and coupling variables for oscillator j
                phase_coupling = self.phase_coupling
                anti_phase_coupling = self.anti_phase_coupling
                other_oscillator_phase = phases[other_oscillator_number]
#
                # phase change of oscillator i due to oscillator j
                between_oscillator_phase = (oscillator_phase - other_oscillator_phase) #% 2 * torch.pi
                phase_difference += torch.as_tensor([- phase_coupling * torch.sin(between_oscillator_phase) - 
                anti_phase_coupling * torch.sin(2 * (between_oscillator_phase))])

        return phase_difference


    def oscillator_system(self, phases):
        """
        Updates phases of the system of 4 mutually influencing oscillators 

        Arguments
        ----------

        phases: torch.tensor of length 4
            phases of the oscillators at the previous timestep


        Returns
        -----------

        phase_differences: torch.tensor of length 4
            phase differences for the 4 oscillators
            i.e. phases(t+1) - phases(t)

        """
        phase_differences = torch.zeros(2)

        # calculate updated phase for each oscillator individually
        for oscillator_number in range(self.number_of_oscillators):
            phase_differences[oscillator_number] = self.single_oscillator(oscillator_number, phases)

        return phase_differences


    def runge_kutta_HKB(self, phases):
        """
        Determine the phases of the oscillators at the current timestep
        using the Runge Kutta method of order 4

        Arguments
        ----------

        phases: torch.tensor of length 4
            phases of the oscillators at the previous timestep


         Returns
        -----------

        new_phases: torch.tensor of length 4
            the phases of the oscillators at the current timestep
          

        """
        k1 = self.oscillator_system(phases) * (1/fs)
        k2 = self.oscillator_system(phases + 0.5 * k1) * (1/fs)
        k3 = self.oscillator_system(phases + 0.5 * k2)  * (1/fs)
        k4 = self.oscillator_system(phases + k3) * (1/fs)
        phase_differences = (1/6) * (k1 + 2 * k2 + 2 * k3 + k4)
        new_phases = phases + phase_differences 
       # new_phases = new_phases % 2 * torch.pi # introduce periodicity

        return new_phases, phase_differences
        

    def next_timestep(self, time, gradient_left, gradient_right, periodic_randomization):
        """"
        Receive input from the environment
        Update the internal states of the agent
        Output the agent's new position and orientation to the environment
        
        Arguments: 
        -----------
            stimulus_intensity_left: float
                stimulus intensity at position of left eye
                stimulus intensity at position of right eye

        Returns:
        ----------
            position (x, y): torch.tensor
                avatar position in world space

            orientation: torch.tensor
                the orientation of the avatar in world space
                proportional to phase difference between motor oscillators


        """
        if periodic_randomization:
            # randomize phases every 20 timesteps (c.f. Aguilera et al., 2013)
            self.phases = self.randomize_phases(time)

        # calculate change in stimulus intensity for each eye
        self.stimulus_change_left = gradient_left
        self.stimulus_change_right = gradient_right


        # determine the new state of the agent
        self.phases, phase_differences  = self.runge_kutta_HKB(self.phases)

        # change in orientation is proportional to phase difference between motor units
        motor_phase_difference = self.phases[0] - self.phases[1] #% 2 * torch.pi 
        #motor_phase_difference = np.exp(self.phases[2] - self.phases[3]) #% 2 * torch.pi 

        motor_phase_difference = np.angle(np.exp(1j * (self.phases[0].detach().numpy() - self.phases[1].detach().numpy()))) #% 2 * torch.pi 
        self.orientation = self.orientation + torch.as_tensor(motor_phase_difference)

        # calculate next position according to movement speed and new orientation
        self.position = self.position + torch.tensor([torch.cos(self.orientation)
         * self.movement_speed * (1/self.fs), torch.sin(self.orientation) * self.movement_speed * (1/self.fs)])

        return self.position, self.orientation, np.array(self.phases), np.array(phase_differences)


    def eye_positions(self):
        """"
        Calculate position of the agent's eyes in world space
        based on the orientation and position
        
        Arguments: 
        -----------
            None; uses variables stored in the class

        Returns:
        ----------
            left_eye_position (x, y): torch.tensor of length 2
                position of the left eye in world space

            right_eye_position (x, y): torch.tensor of length 2
                position of the right eye in world space

        """
        left_eye_position = torch.zeros(2)
        right_eye_position = torch.zeros(2)

        left_eye_position[0] = self.position[0] + torch.sin(self.orientation - self.agent_eye_angle/2 ) * self.agent_radius
        left_eye_position[1] = self.position[1] + torch.cos(self.orientation - self.agent_eye_angle/2 ) * self.agent_radius

        right_eye_position[0] = self.position[0] + torch.sin(self.orientation + self.agent_eye_angle/2 ) * self.agent_radius
        right_eye_position[1] = self.position[1] + torch.cos(self.orientation + self.agent_eye_angle/2 ) * self.agent_radius

        return left_eye_position, right_eye_position 


    def randomize_phases(self, time):
        """
        Takes in the current phases of each oscillator and
        assigns random values to them from a uniform distribution
        between 0 and 2pi

        """
        if time % 20 == 0:
            self.phases = 2 * torch.pi * torch.rand(len(self.phases))

        return self.phases
 