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

    def __init__(self, fs, duration, stimulus_positions, stimulus_ratio, stimulus_decay_rate,
     stimulus_scale, stimulus_sensitivity, starting_position, starting_orientation, movement_speed, agent_radius, agent_eye_angle, delta_orientation):
        self.fs = fs
        self.duration = duration
        self.stimulus_positions = stimulus_positions # list of 2-element arrays
        self.stimulus_ratio = stimulus_ratio
        self.stimulus_decay_rate = stimulus_decay_rate
        self.stimulus_scale = stimulus_scale
        self.stimulus_sensitivity = stimulus_sensitivity
        self.movement_speed = movement_speed
        self.agent_radius = agent_radius
        self.agent_eye_angle = agent_eye_angle
        self.delta_orientation = delta_orientation

        self.reset(starting_position, starting_orientation)



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

        # the first (left) stimulus is largest if the ratio is less than 1
        # i.e. the ratio = stimulus_strenght_left / stimulus_strength_right
        self.correct_position = self.stimulus_positions[0]
        if len(self.stimulus_positions) > 1:
            # if two stimulus
            if self.stimulus_ratio > 1:
                # if the right stimulus is larger
                self.correct_position  = self.stimulus_positions[1]



        self.distance = eucl_distance_np(self.correct_position, self.position)
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
        

        output_angle =  np.angle(np.exp(1j*(action)))



       # orientation += output_angle  #np.sin(action)*self.delta_orientation / self.fs
        self.orientation += 50 * output_angle / self.fs 
        #self.orientation = output_angle 


        self.position = self.position + np.array([np.sin(self.orientation)
        * self.movement_speed * (1/self.fs), np.cos(self.orientation) * self.movement_speed * (1/self.fs)])


        # calculate next position according to movement speed and new orientation
        
        self.position_x.append(self.position[0])
        self.position_y.append(self.position[1])

        # get new state and reward
        left_eye_position, right_eye_position = self.eye_positions()



        # get gradient directly

        left_gradient = self.get_stimulus_concentration(left_eye_position)
        right_gradient = self.get_stimulus_concentration(right_eye_position)

        # print("left gradient" + str(left_gradient))
        # print("right gradient" + str(right_gradient))

        # the agent will observe the stimulus gradient at its eyes (state)
        state = self.stimulus_sensitivity  * np.array([left_gradient, right_gradient]) #* self.fs
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
        distances = []
        for stimulus_position in self.stimulus_positions:
            distances.append(eucl_distance_np(stimulus_position, self.position))
        self.distance = np.min(distances)

        if self.distance < 10:
            reward = (self.duration - self.time) * self.stimulus_scale 
            #done = True

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
        distances = []
        for stimulus_position in self.stimulus_positions:
            distances.append(eucl_distance_np(stimulus_position, location))

        stimulus_concentration =  self.stimulus_scale * np.exp( - self.stimulus_decay_rate * distances[0])
        if len(distances) > 1:
            stimulus_concentration_1 =  self.stimulus_scale * np.exp( - self.stimulus_decay_rate * distances[0])
            stimulus_concentration_2 = self.stimulus_scale * np.exp( - self.stimulus_decay_rate * distances[1])
            stimulus_concentration = stimulus_concentration_1 + self.stimulus_ratio * stimulus_concentration_2

        return stimulus_concentration


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

        distances = []
        for stimulus_position in self.stimulus_positions:
            # distance to stimuli center
            distances.append(eucl_distance_np(stimulus_position, location))

        stimulus_gradient =  self.stimulus_decay_rate * self.stimulus_scale * np.exp( - self.stimulus_decay_rate * distances[0])
        if len(distances) > 1:
            # if moer than one stimulus
            stimulus_gradient_1 =  self.stimulus_decay_rate * self.stimulus_scale * np.exp( - self.stimulus_decay_rate * distances[0])
            stimulus_gradient_2 = self.stimulus_decay_rate * self.stimulus_scale * np.exp( - self.stimulus_decay_rate * distances[1])
            stimulus_gradient = stimulus_gradient_1 + self.stimulus_ratio * stimulus_gradient_2

        return stimulus_gradient
        #return self.stimulus_decay_rate * self.stimulus_scale * np.exp( - self.stimulus_decay_rate * self.distance)


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

    def __init__(self, fs, duration, stimulus_positions, stimulus_decay_rate,
     stimulus_scale, stimulus_sensitivity, movement_speed, agent_radius, agent_eye_angle, delta_orientation, stimulus_ratio, n_agents):
        """
        starting_positions = list with one tuple per agent
        """
        self.fs = fs
        self.duration = duration
        self.stimulus_positions = stimulus_positions
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
        self.n_agents = n_agents
        self.stimulus_ratio = stimulus_ratio

    def reset(self, starting_positions, starting_orientations, n_agents):
        """ For the next episode, train the angent in the same 
        environment but with a different initial position and orientation"""
        self.agent_positions = starting_positions
        self.agent_new_positions = starting_positions
        self.agent_orientations = starting_orientations
        self.agent_new_orientations = starting_orientations
        self.right_stimulus_intensity = 0
        self.left_stimulus_intensity = 0
        self.time = 0
        
        self.position_x = []
        self.position_y = []
        self.save_orientations = []
        self.distances = []

        # the first (left) stimulus is largest if the ratio is less than 1
        # i.e. the ratio = stimulus_strenght_left / stimulus_strength_right
        self.correct_position = self.stimulus_positions[0]
        if len(self.stimulus_positions) > 1:
            # if two stimulus
            if self.stimulus_ratio > 1:
                # if the right stimulus is larger
                self.correct_position  = self.stimulus_positions[1]



        self.inter_agent_distances = np.zeros((n_agents, n_agents))
        states = []
        for i in range(n_agents): 
            self.position_x.append([])
            self.position_y.append([])
            self.save_orientations.append([])
            self.distances.append(eucl_distance_np(self.correct_position, self.agent_positions[i]))
            states.append(np.array([self.left_stimulus_intensity, self.right_stimulus_intensity]))
        return states

    def step(self, actions, food_size):
        """action is moving right, moving left or continuing going forward """
        states = []
        rewards = []
        distances = []


        # update the distances between the agents
        for i in range(self.n_agents):
            for j in range(self.n_agents):
                if i != j:
                    inter_agent_distance = eucl_distance_np(self.agent_positions[i], self.agent_positions[j])
                    self.inter_agent_distances[i, j] = inter_agent_distance


        # loop through all Guidos
        for i in range(self.n_agents):
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
            output_angle = np.angle(np.exp(1j*(action)))
            orientation += 25 * output_angle / self.fs 
            self.agent_new_orientations[i] = orientation  #% (2 * np.pi)
            # calculate next position according to movement speed and new orientation
            self.agent_new_positions[i] = np.array(position) + np.array([np.sin(orientation)
            * self.movement_speed * (1/self.fs), np.cos(orientation) * self.movement_speed * (1/self.fs)])



            self.position_x[i].append(position[0])
            self.position_y[i].append(position[1])
            self.save_orientations[i].append(orientation)

            # get new state and reward
            left_eye_position, right_eye_position = self.eye_positions(self.agent_new_positions[i], self.agent_new_orientations[i])
            



            # get gradient directly
            left_gradient = self.get_stimulus_concentration(left_eye_position)
            right_gradient = self.get_stimulus_concentration(right_eye_position)



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

            distances = []
            for stimulus_position in self.stimulus_positions:
                distances.append(eucl_distance_np(stimulus_position, self.agent_new_positions[i]))
            distance = np.min(distances)

            self.distances[i] = distance
            if distance < 10:
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
        distances = []
        for stimulus_position in self.stimulus_positions:
            distances.append(eucl_distance_np(stimulus_position, location))

        stimulus_concentration =  self.stimulus_scale * np.exp( - self.stimulus_decay_rate * distances[0])
        if len(distances) > 1:
            stimulus_concentration_1 =  self.stimulus_scale * np.exp( - self.stimulus_decay_rate * distances[0])
            stimulus_concentration_2 = self.stimulus_scale * np.exp( - self.stimulus_decay_rate * distances[1])
            stimulus_concentration = stimulus_concentration_1 + self.stimulus_ratio * stimulus_concentration_2

        return stimulus_concentration

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

        distances = []
        for stimulus_position in self.stimulus_positions:
            # distance to stimuli center
            distances.append(eucl_distance_np(stimulus_position, location))

        stimulus_gradient =  self.stimulus_decay_rate * self.stimulus_scale * np.exp( - self.stimulus_decay_rate * distances[0])
        if len(distances) > 1:
            # if moer than one stimulus
            stimulus_gradient_1 =  self.stimulus_decay_rate * self.stimulus_scale * np.exp( - self.stimulus_decay_rate * distances[0])
            stimulus_gradient_2 = self.stimulus_decay_rate * self.stimulus_scale * np.exp( - self.stimulus_decay_rate * distances[1])
            stimulus_gradient = stimulus_gradient_1 + self.stimulus_ratio * stimulus_gradient_2

        return stimulus_gradient
        #return self.stimulus_decay_rate * self.stimulus_scale * np.exp( - self.stimulus_decay_rate * self.distance)



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


    
class Social_stimulus_environment():

    def __init__(self, fs, duration, stimulus_positions, stimulus_decay_rate,
     stimulus_scale, stimulus_sensitivity, movement_speed, agent_radius, agent_eye_angle, delta_orientation, agent_stimulus_scale, agent_stimulus_decay_rate, stimulus_ratio, n_agents):
        """
        starting_positions = list with one tuple per agent
        """
        self.fs = fs
        self.duration = duration
        self.stimulus_positions = stimulus_positions
        self.stimulus_decay_rate = stimulus_decay_rate
        self.stimulus_scale = stimulus_scale
        
        self.agent_stimulus_scale = agent_stimulus_scale
        self.agent_stimulus_decay_rate = agent_stimulus_decay_rate

        self.stimulus_sensitivity = stimulus_sensitivity
        self.movement_speed = movement_speed
        self.agent_radius = agent_radius
        self.agent_eye_angle = agent_eye_angle
        self.right_stimulus_intensity = 0
        self.left_stimulus_intensity = 0
        self.delta_orientation = delta_orientation
        self.time = 0
        self.n_agents = n_agents
        self.stimulus_ratio = stimulus_ratio

    def reset(self, starting_positions, starting_orientations, n_agents):
        """ For the next episode, train the angent in the same 
        environment but with a different initial position and orientation"""
        self.agent_positions = starting_positions
        self.agent_new_positions = starting_positions
        self.agent_orientations = starting_orientations
        self.agent_new_orientations = starting_orientations
        self.right_stimulus_intensity = 0
        self.left_stimulus_intensity = 0
        self.time = 0
        
        self.position_x = []
        self.position_y = []
        self.save_orientations = []
        self.distances = []


        # the first (left) stimulus is largest if the ratio is less than 1
        # i.e. the ratio = stimulus_strenght_left / stimulus_strength_right
        self.correct_position = self.stimulus_positions[0]
        if len(self.stimulus_positions) > 1:
            # if two stimulus
            if self.stimulus_ratio > 1:
                # if the right stimulus is larger
                self.correct_position  = self.stimulus_positions[1]



        self.inter_agent_distances = np.zeros((n_agents, n_agents))
        states = []
        for i in range(n_agents): 
            self.position_x.append([])
            self.position_y.append([])
            self.save_orientations.append([])
            self.distances.append(eucl_distance_np(self.correct_position, self.agent_positions[i]))
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
            output_angle = np.angle(np.exp(1j*(action)))
            orientation += 25 * output_angle / self.fs 
            self.agent_new_orientations[i] = orientation  #% (2 * np.pi)
            # calculate next position according to movement speed and new orientation
            self.agent_new_positions[i] = np.array(position) + np.array([np.sin(orientation)
            * self.movement_speed * (1/self.fs), np.cos(orientation) * self.movement_speed * (1/self.fs)])




            self.position_x[i].append(position[0])
            self.position_y[i].append(position[1])
            self.save_orientations[i].append(orientation)

            # get new state and reward
            left_eye_position, right_eye_position = self.eye_positions(self.agent_new_positions[i], self.agent_new_orientations[i])
            



            # get gradient directly
            left_gradient = self.get_stimulus_concentration(left_eye_position)
            right_gradient = self.get_stimulus_concentration(right_eye_position)
            
            # get gradient due to agents
            left_agent_gradient = self.get_agent_concentration(left_eye_position, i)
            right_agent_gradient = self.get_agent_concentration(right_eye_position, i)

            # merge the two gradients together
            left_gradient += left_agent_gradient
            right_gradient += right_agent_gradient

            
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


            distances = []
            for stimulus_position in self.stimulus_positions:
                distances.append(eucl_distance_np(stimulus_position, self.agent_new_positions[i]))
            distance = np.min(distances)

            self.distances[i] = distance
            if distance < 10:
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
        distances = []
        for stimulus_position in self.stimulus_positions:
            distances.append(eucl_distance_np(stimulus_position, location))

        stimulus_concentration =  self.stimulus_scale * np.exp( - self.stimulus_decay_rate * distances[0])
        if len(distances) > 1:
            stimulus_concentration_1 =  self.stimulus_scale * np.exp( - self.stimulus_decay_rate * distances[0])
            stimulus_concentration_2 = self.stimulus_scale * np.exp( - self.stimulus_decay_rate * distances[1])
            stimulus_concentration = stimulus_concentration_1 + self.stimulus_ratio * stimulus_concentration_2

        return stimulus_concentration

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

        distances = []
        for stimulus_position in self.stimulus_positions:
            # distance to stimuli center
            distances.append(eucl_distance_np(stimulus_position, location))

        stimulus_gradient =  self.stimulus_decay_rate * self.stimulus_scale * np.exp( - self.stimulus_decay_rate * distances[0])
        if len(distances) > 1:
            # if moer than one stimulus
            stimulus_gradient_1 =  self.stimulus_decay_rate * self.stimulus_scale * np.exp( - self.stimulus_decay_rate * distances[0])
            stimulus_gradient_2 = self.stimulus_decay_rate * self.stimulus_scale * np.exp( - self.stimulus_decay_rate * distances[1])
            stimulus_gradient = stimulus_gradient_1 + self.stimulus_ratio * stimulus_gradient_2

        return stimulus_gradient
        #return self.stimulus_decay_rate * self.stimulus_scale * np.exp( - self.stimulus_decay_rate * self.distance)


    def get_agent_concentration(self, location, i):
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
        agent_gradient = 0
        for a in range(len(self.agent_new_positions)):
            if a != i:
                distance = eucl_distance_np(self.agent_new_positions[a], location)
                agent_gradient += self.agent_stimulus_scale * np.exp( - self.agent_stimulus_decay_rate * distance)
        return agent_gradient


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




