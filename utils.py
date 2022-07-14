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

def symmetric_matrix_5_torch(a):

    return torch.tensor([[0, a, a, a, a],
                        [a, 0, a, a, a],
                        [a, a, 0, a, a],
                        [a, a, a, 0, a],
                        [a, a, a, a, 0]])

def symmetric_matrix_4_torch(a):

    return torch.tensor([[0, a, a, a, a],
                        [a, 0, a, a, a],
                        [a, a, 0, a, a],
                        [a, a, a, 0, a]])

def eucl_distance(location_1, location_2):

    return torch.sqrt(torch.pow(location_1[0] - location_2[0],2) + torch.pow(location_1[1] - location_2[1],2))

