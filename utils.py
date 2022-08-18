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

