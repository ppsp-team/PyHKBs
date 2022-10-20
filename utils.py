#!/usr/bin/env python
# coding=utf-8
# ==============================================================================
# title           : utils.py
# description     : helper functions
# author          : Nicolas Coucke
# date            : 2022-07-11
# version         : 1
# usage           : python utils.py
# notes           : install the packages with "pip install -r requirements.txt"
# python_version  : 3.9.2
# ==============================================================================


import torch 
import numpy as np

def symmetric_matrix(value, size):
    """
    Helper function to create coupling matrix with equal weights between oscillator
    The "self-coupling weight" (the diagonal) is set to zero

    Arguments:
    ---------
    value: float
        the coupling value between oscillators
    size: int
        the size of the matrix (size * size)

    """
    matrix = value * torch.ones((size, size))
    matrix.fill_diagonal_(0)

    return matrix


def eucl_distance(location_1, location_2):
    """
    Helper function to calculate the euclidian distance between two points

    Arguments: 
    ---------
    location_1: tensor of length 2
        the x and y coordinates of the first point

    location_2: idem

    Returns:
    --------
    Distance 

    """
    return torch.sqrt(torch.pow(location_1[0] - location_2[0],2) + torch.pow(location_1[1] - location_2[1],2))

def eucl_distance_np(location_1, location_2):
    """
    Helper function to calculate the euclidian distance between two points

    Arguments: 
    ---------
    location_1: tensor of length 2
        the x and y coordinates of the first point

    location_2: idem

    Returns:
    --------
    Distance 

    """
    return np.sqrt((location_1[0] - location_2[0])**2 + (location_1[1] - location_2[1])**2)


def complementary_connection(connection_strength, asymmetry_degree):
   return (1 - connection_strength) * asymmetry_degree + (1 - asymmetry_degree) * connection_strength


def initiate_coupling_weights(scale, asymmetry_degree, random_connections, n_oscillators):

    if random_connections: 
        a_sens = np.random.uniform()
        a_motor = np.random.uniform()
        a_ips_left = np.random.uniform()
        a_con_left = np.random.uniform()
    else:
        a_sens = 1
        a_motor = 1
        a_ips_left = 1
        a_con_left = 1

   # make random variables for connectivities:
    a_ips_right = complementary_connection(a_ips_left, asymmetry_degree)
    a_con_right = complementary_connection(a_con_left, asymmetry_degree)


    if n_oscillators == 5:
        # also determine connections to the 5th oscillator
        if random_connections:
            a_soc_sens_left = np.random.uniform()
            a_soc_motor_left = np.random.uniform()
        else: 
            a_soc_sens_left = 1
            a_soc_motor_left = 1

        a_soc_sens_right = complementary_connection(a_soc_sens_left, asymmetry_degree)
        a_soc_motor_right = complementary_connection(a_soc_motor_left, asymmetry_degree)

        coupling_weights = scale * np.array([ a_sens, a_ips_left, a_ips_right, a_con_left, a_con_right, a_motor,
                    a_soc_sens_left, a_soc_sens_right, a_soc_motor_left, a_soc_motor_right])
        return coupling_weights, a_sens, a_ips_left, a_ips_right, a_con_left, a_con_right, a_motor, a_soc_sens_left, a_soc_sens_right, a_soc_motor_left, a_soc_motor_right

    else:
        coupling_weights = scale * np.array([ a_sens, a_ips_left, a_ips_right, a_con_left, a_con_right, a_motor])

        return coupling_weights, a_sens, a_ips_left, a_ips_right, a_con_left, a_con_right, a_motor
