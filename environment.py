#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : environment.py
# description     : contains environment class to train agents
# author          : Nicolas Coucke
# date            : 2022-08-16
# version         : 1
# usage           : use within training_RL.py
# notes           : install the packages with "pip install -r requirements.txt"
# python_version  : 3.9.2
# ==============================================================================

import matplotlib.pyplot as plt
import torch
import numpy as np
from utils import eucl_distance, symmetric_matrix, eucl_distance_np
from agent import Agent
import time
from matplotlib import animation
import tkinter as tk
import cmath



# during the training: vary starting position and orientation
# In this function, the agent's body (position etc) is part of the environment, 
# when spawning multiple agents, this has to be adapted to accomodate multiple agents 
# in which each agent has a class within the environment

class Environment():

    def __init__(self, fs, duration, stimulus_position, stimulus_decay_rate,
     stimulus_scale, stimulus_sensitivity, starting_position, starting_orientation, movement_speed, agent_radius, agent_eye_angle, delta_orientation):
        self.fs = fs
        self.duration = duration
        self.stimulus_position = stimulus_position
        self.stimulus_decay_rate = stimulus_decay_rate
        self.stimulus_scale = stimulus_scale
        self.stimulus_sensitivity = stimulus_sensitivity
        self.position = starting_position
        self.orientation = starting_orientation
        self.movement_speed = movement_speed
        self.agent_radius = agent_radius
        self.agent_eye_angle = agent_eye_angle
        self.right_stimulus_intensity = 0
        self.left_stimulus_intensity = 0
        self.delta_orientation = delta_orientation
        self.time = 0



    def reset(self, starting_position, starting_orientation):
        """ For the next episode, train the angent in the same 
        environment but with a different initial position and orientation"""
        self.position = starting_position
        self.orientation = starting_orientation
        self.right_stimulus_intensity = 0
        self.left_stimulus_intensity = 0
        self.time = 0
        self.position_x = []
        self.position_y = []
        self.distance = eucl_distance_np(self.stimulus_position, self.position)
        return np.array([self.left_stimulus_intensity, self.right_stimulus_intensity])

    def step(self, action, food_size):
        """action is moving right, moving left or continuing going forward """

        # execute action
        if action == 0:
            # turn right
            self.orientation = self.orientation + self.delta_orientation / self.fs
        elif action == 1:
            # turn left
            self.orientation = self.orientation - self.delta_orientation / self.fs
        #elif action == 2:
        #   keep moving forward

        self.orientation = self.orientation + action #np.sin(action)*self.delta_orientation / self.fs
        self.position = self.position + np.array([np.sin(self.orientation)
        * self.movement_speed * (1/self.fs), np.cos(self.orientation) * self.movement_speed * (1/self.fs)])


        # calculate next position according to movement speed and new orientation
        
        self.position_x.append(self.position[0])
        self.position_y.append(self.position[1])

        # get new state and reward
        left_eye_position, right_eye_position = self.eye_positions()


        new_left_stimulus_intensity = self.get_stimulus_concentration(left_eye_position)
        new_right_stimulus_intensity = self.get_stimulus_concentration(right_eye_position)


        # get the difference between previous and current stimulus intensity
        left_gradient = new_left_stimulus_intensity - self.left_stimulus_intensity
        right_gradient = new_right_stimulus_intensity - self.right_stimulus_intensity

        # save the current intensity for the next iteration
        self.left_stimulus_intensity = new_left_stimulus_intensity
        self.right_stimulus_intensity = new_right_stimulus_intensity

        # get gradient directly
        left_gradient = self.get_stimulus_gradient(left_eye_position)
        right_gradient = self.get_stimulus_gradient(right_eye_position)

        # the agent will observe the stimulus gradient at its eyes (state)
        state = self.stimulus_sensitivity  * np.array([left_gradient, right_gradient]) * self.fs
       # state = self.stimulus_sensitivity * np.array([new_left_stimulus_intensity, new_right_stimulus_intensity])


        # the food is the stimulus concentration at the center of the body
        food = self.get_stimulus_concentration(self.position)

        # punish agent for staying too long away from the food
        hunger = 2

        # reward is a combination of food and funger
        reward = food - hunger

        # end the episode when the time taken is too long
        if self.time > self.duration:
            done = True
        else:
            done = False

        # or when the agent has found the food source
        self.distance = eucl_distance_np(self.stimulus_position, self.position)
        if self.distance < food_size:
            reward = (self.duration - self.time) * self.stimulus_scale 
           # done = True

        self.time += 1/self.fs
        return state, reward, done


    def get_stimulus_concentration(self, location):
        """
        Get the concentration of the stimulus at a certain location

        Arguments:
        ----------
        location: numpy array of length x
            [x position, y position]

        Returns:
        ----------
        stimulus_concentration: float

        """
        self.distance = eucl_distance_np(self.stimulus_position, location)
        return  self.stimulus_scale * np.exp( - self.stimulus_decay_rate * self.distance)


    def get_stimulus_gradient(self, location):
        """
        Get the concentration of the stimulus at a certain location

        Arguments:
        ----------
        location: numpy array of length x
            [x position, y position]

        Returns:
        ----------
        stimulus_concentration: float

        """
        self.distance = eucl_distance_np(self.stimulus_position, location)
        return self.stimulus_decay_rate * self.stimulus_scale * np.exp( - self.stimulus_decay_rate * self.distance)


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
        left_eye_position = np.zeros(2)
        right_eye_position = np.zeros(2)

        left_eye_position[0] = self.position[0] + np.sin(self.orientation - self.agent_eye_angle/2 ) * self.agent_radius
        left_eye_position[1] = self.position[1] + np.cos(self.orientation - self.agent_eye_angle/2 ) * self.agent_radius

        right_eye_position[0] = self.position[0] + np.sin(self.orientation + self.agent_eye_angle/2 ) * self.agent_radius
        right_eye_position[1] = self.position[1] + np.cos(self.orientation + self.agent_eye_angle/2 ) * self.agent_radius

        return left_eye_position, right_eye_position 





class Social_environment():

    def __init__(self, fs, duration, stimulus_position, stimulus_decay_rate,
     stimulus_scale, stimulus_sensitivity, starting_positions, starting_orientations, movement_speed, agent_radius, agent_eye_angle, delta_orientation):
        """
        starting_positions = list with one tuple per agent
        """
        self.fs = fs
        self.duration = duration
        self.stimulus_position = stimulus_position
        self.stimulus_decay_rate = stimulus_decay_rate
        self.stimulus_scale = stimulus_scale
        self.stimulus_sensitivity = stimulus_sensitivity
        self.movement_speed = movement_speed
        self.agent_radius = agent_radius
        self.agent_eye_angle = agent_eye_angle
        self.right_stimulus_intensity = 0
        self.left_stimulus_intensity = 0
        self.delta_orientation = delta_orientation
        self.time = 0

        self.agent_positions = starting_positions
        self.agent_orientations = starting_orientations
        
        # you have to store and not replace the old ones right away so that you can update all of them at the same time
        self.agent_new_positions = starting_positions
        self.agent_new_orientations = starting_orientations

    def reset(self, starting_positions, starting_orientations, n_agents):
        """ For the next episode, train the angent in the same 
        environment but with a different initial position and orientation"""
        self.agent_positions = starting_positions
        self.agent_new_positions = starting_positions
        self.agent_orentation = starting_orientations
        self.right_stimulus_intensity = 0
        self.left_stimulus_intensity = 0
        self.time = 0
        
        self.position_x = []
        self.position_y = []
        self.distances = []

        states = []
        for i in range(n_agents):
            new_list = []
            self.position_x.append(new_list)
            self.position_y.append(new_list)
            self.distances.append(eucl_distance_np(self.stimulus_position, self.agent_positions[i]))
            states.append(np.array([self.left_stimulus_intensity, self.right_stimulus_intensity]))
        return states

    def step(self, actions, food_size):
        """action is moving right, moving left or continuing going forward """
        states = []
        rewards = []
        distances = []

        # loop through all Guidos
        for i in range(len(actions)):
            action = actions[i]
            orientation = self.agent_orientations[i]
            position = self.agent_positions[i]
            # execute action
            if action == 0:
                # turn right
                self.agent_new_orientations[i] = orientation + self.delta_orientation / self.fs
            elif action == 1:
                # turn left
                self.agent_new_orientations[i] = orientation - self.delta_orientation / self.fs
            #elif action == 2:
            #   keep moving forward
            
            # new version: do the gradual 
            orientation +=  action #np.sin(action)*self.delta_orientation / self.fs
            self.agent_new_orientations[i] = orientation
            # calculate next position according to movement speed and new orientation
            self.agent_new_positions[i] = np.array(position) + np.array([np.sin(orientation)
            * self.movement_speed * (1/self.fs), np.cos(orientation) * self.movement_speed * (1/self.fs)])

            self.position_x[i].append(position[0])
            self.position_y[i].append(position[1])

            # get new state and reward
            left_eye_position, right_eye_position = self.eye_positions(self.agent_new_positions[i], self.agent_new_orientations[i])
            



            # get gradient directly
            left_gradient = self.get_stimulus_gradient(left_eye_position)
            right_gradient = self.get_stimulus_gradient(right_eye_position)

        
            # the agent will observe the stimulus gradient at its eyes (state)
            state = self.stimulus_sensitivity  * np.array([left_gradient, right_gradient]) #* self.fs
            # state = self.stimulus_sensitivity * np.array([new_left_stimulus_intensity, new_right_stimulus_intensity])
            states.append(state)

            # the food is the stimulus concentration at the center of the body
            food = self.get_stimulus_concentration(position)

            # punish agent for staying too long away from the food
            hunger = 2

            # reward is a combination of food and funger
            reward = food - hunger
            rewards.append(reward)
            # end the episode when the time taken is too long
            if self.time > self.duration:
                done = True
            else:
                done = False

            # or when the agent has found the food source
            distance = eucl_distance_np(self.stimulus_position, self.agent_new_positions[i])
            self.distances[i] = distance
            if distance < food_size:
                reward = (self.duration - self.time) * self.stimulus_scale 

            # done = True


        # after having calculated the new position and angle for each agent, update them
        self.agent_positions = self.agent_new_positions
        self.agent_orientations = self.agent_new_orientations

        self.time += 1/self.fs

        return states, rewards, done


    def get_stimulus_concentration(self, location):
        """
        Get the concentration of the stimulus at a certain location

        Arguments:
        ----------
        location: numpy array of length x
            [x position, y position]

        Returns:
        ----------
        stimulus_concentration: float

        """
        self.distance = eucl_distance_np(self.stimulus_position, location)
        return  self.stimulus_scale * np.exp( - self.stimulus_decay_rate * self.distance)


    def get_stimulus_gradient(self, location):
        """
        Get the concentration of the stimulus at a certain location

        Arguments:
        ----------
        location: numpy array of length x
            [x position, y position]

        Returns:
        ----------
        stimulus_concentration: float

        """
        self.distance = eucl_distance_np(self.stimulus_position, location)
        return self.stimulus_decay_rate * self.stimulus_scale * np.exp( - self.stimulus_decay_rate * self.distance)


    def eye_positions(self, position, orientation):
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
        left_eye_position = np.zeros(2)
        right_eye_position = np.zeros(2)

        left_eye_position[0] = position[0] + np.sin(orientation - self.agent_eye_angle/2 ) * self.agent_radius
        left_eye_position[1] = position[1] + np.cos(orientation - self.agent_eye_angle/2 ) * self.agent_radius

        right_eye_position[0] = position[0] + np.sin(orientation + self.agent_eye_angle/2 ) * self.agent_radius
        right_eye_position[1] = position[1] + np.cos(orientation + self.agent_eye_angle/2 ) * self.agent_radius

        return left_eye_position, right_eye_position 

