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

# define sampling frequency
fs = torch.as_tensor([100])  # Hertz


class Agent:
    def __init__(self, agent_id, stimulus_sensitivity, phase_coupling_matrix, anti_phase_coupling_matrix, initial_phases, frequencies, movement_speed):

        # define agent variables that stay constant during the simulation
        self.id = agent_id
        self.phase_coupling_matrix = phase_coupling_matrix
        self.anti_phase_coupling_matrix = anti_phase_coupling_matrix
        self.initial_phases = initial_phases
        self.frequencies = frequencies
        self.stimulus_sensitivity = stimulus_sensitivity
        self.movement_speed = movement_speed
        self.number_of_oscillators = 4
        self.agent_radius = torch.as_tensor([1.])
        self.agent_eye_angle = torch.pi/4  # 45 degree angle between the eyes of the agent

        # initialize agent variables that change during the simulation
        self.phases = initial_phases
        self.stimulus_intensity_left = torch.as_tensor([0.])
        self.stimulus_intensity_right = torch.as_tensor([0.])
        self.stimulus_change_left = torch.as_tensor([0.])
        self.stimulus_change_right = torch.as_tensor([0.])
        self.position = torch.as_tensor([0., 0.])
        self.orientation = torch.as_tensor([0.])


    def single_oscillator(self, oscillator_number, phases):
        """"
        The phase of a oscillator i is modified by being connected to the other oscillator j by means of the HKB equations
        If the oscillator is part of the sensory system, then it's phase is also influenced by the change in stimulus intensity

        Arguments
        ----------

        oscillator number: int
            1, 2, 3 or 4

        phases: torch.tensor or length 4
            the phase of each oscillator at the previous timestep

        Returns
        --------

        phase_difference: torch.tensor of length 1
            phase change of oscillator i

        """

        # phase of oscillator 
        oscillator_phase = phases[oscillator_number]

        # phase change of oscillator i due to intrinsic frequency
        phase_difference = torch.as_tensor([2 * torch.pi * self.frequencies[oscillator_number]])

        # include sensory input for the eyes
        if oscillator_number == 0: # left eye
            phase_difference += self.stimulus_sensitivity*self.stimulus_change_left
        elif oscillator_number == 1: # right eye
            phase_difference += self.stimulus_sensitivity*self.stimulus_change_right

        for other_oscillator_number in range(4): # loop through all other oscillators

            if other_oscillator_number != oscillator_number:

                # get the phase and coupling variables for oscillator j
                phase_coupling = self.phase_coupling_matrix[oscillator_number,
                    other_oscillator_number]
                anti_phase_coupling = self.anti_phase_coupling_matrix[oscillator_number, 
                    other_oscillator_number]
                other_oscillator_phase = phases[other_oscillator_number]

                # phase change of oscillator i due to oscillator j
                phase_difference += torch.as_tensor([- phase_coupling * torch.sin(oscillator_phase
                    - other_oscillator_phase) - anti_phase_coupling * torch.sin(2 * (oscillator_phase - other_oscillator_phase))])

        return phase_difference


    def oscillator_system(self, phases):
        """
        Describes the system of N mutually influencing oscillators 

        Arguments
        ----------

        phases: torch.tensor of length 4
            phases of the oscillators at the previous timestep


        Returns
        -----------

        phase_differences: torch.tensor of length 4
            phases difference for the 4 oscillators

        """
        phase_differences = torch.as_tensor(np.zeros(4,))

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
        new_phases = phases + (1/6) * (k1 + 2 * k2 + 2 * k3 + k4)

        return new_phases
        

    def next_timestep(self, stimulus_intensity_left, stimulus_intensity_right):
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
        # calculate change in stimulus intensity for each eye
        self.stimulus_change_left = stimulus_intensity_left - self.stimulus_intensity_left
        self.stimulus_change_right = stimulus_intensity_right - self.stimulus_intensity_right

        # save intensity for next iteration
        self.stimulus_intensity_left = stimulus_intensity_left
        self.stimulus_intensity_right = stimulus_intensity_right

        # determine the new state of the agent
        self.phases = self.runge_kutta_HKB(self.phases)

        # change in orientation is proportional to phase difference between motor units
        self.orientation = self.orientation + (self.phases[2] - self.phases[3])

        # calculate next position according to movement speed and new orientation
        self.position = self.position + torch.tensor([torch.cos(self.orientation)
         * self.movement_speed * (1/fs), torch.sin(self.orientation) * self.movement_speed * (1/fs)]) # ypos

        return self.position, self.orientation


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
        left_eye_position = torch.tensor([0., 0.])
        right_eye_position = torch.tensor([0., 0.])

        left_eye_position[0] = self.position[0] + torch.cos(self.orientation - self.agent_eye_angle/2 )*self.agent_radius
        left_eye_position[1] = self.position[1] + torch.sin(self.orientation - self.agent_eye_angle/2 )*self.agent_radius

        right_eye_position[0] = self.position[0] + torch.cos(self.orientation + self.agent_eye_angle/2 )*self.agent_radius
        right_eye_position[1] = self.position[1] + torch.sin(self.orientation + self.agent_eye_angle/2 )*self.agent_radius

        return left_eye_position, right_eye_position 
