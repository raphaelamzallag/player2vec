"""
This module contains several utils functions for processing StatsBomb data.
"""

import ast
import operator
import os

import numpy as np
from player2vec.params import CONSTANTS

def get_location_bin(x,y,dim_x=120,dim_y=80,num_bins_x=5,num_bins_y=5):
    bin_size_x = dim_x/num_bins_x
    bin_size_y = dim_y/num_bins_y
    lst_x = []
    lst_y = []
    for i in range(0,num_bins_x):
        lst_x.append(bin_size_x)
        bin_size_x += dim_x/num_bins_x
    for i in range(0,num_bins_y):
        lst_y.append(bin_size_y)
        bin_size_y += dim_y/num_bins_y
    for i in range(len(lst_x)):
        if x <= lst_x[i]:
            x = f'{i+1}/5'
            break
    for j in range(len(lst_y)):
        if y <= lst_y[j]:
            y = f'{j+1}/5'
            break
    return f'({x}, {y})'
