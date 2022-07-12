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

def symmetric_matrix_torch(a):

    return torch.tensor([[0, a, a, a, a],
                        [a, 0, a, a, a],
                        [a, a, 0, a, a],
                        [a, a, a, 0, a],
                        [a, a, a, a, 0]])


