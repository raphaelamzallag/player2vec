"""
This module contains several utils functions for processing StatsBomb data.
"""

import ast
import operator
import os

import numpy as np
from player2vec.params import CONSTANTS


def get_location_bin(x, y, pitch_dimensions=CONSTANTS.PITCH_DIMENSIONS, output='bin_rel',
                     num_x_bins: int = 5, num_y_bins: int = 5, rel_coordinates=True):
    '''
    :param x: float, x value [-pitch_dimensions[0] / 2, pitch_dimensions[0] / 2]
    :param y: float, y value [-pitch_dimensions[1] / 2, pitch_dimensions[1] / 2]
    :param pitch_dimensions: (x, y) of pitch dimensions
    :param num_x_bins: number of bins to split the length of the pitch (along pitch_dimensions[0])
    :param num_y_bins: number of bins to split the width of the pitch (along pitch_dimensions[1])
    :param rel_coordinates: if True, coordinates assumed to be relative to pitch center
    :param output: 'bin_name', 'bin_rel', or 'bin_ix'
    :return:
    '''

    bin_names = {'x': {3: ['back', 'med', 'fwd'],
                       4: ['back', 'mback', 'mfwd', 'fwd']},
                 'y': {3: ['left', 'center', 'right'],
                       4: ['left', 'mleft', 'mright', 'right'],
                       5: ['left', 'mleft', 'enter', 'mright', 'right']}}

    bin_x_width, bin_y_width = np.ceil(pitch_dimensions[0] / num_x_bins), np.ceil(pitch_dimensions[1] / num_y_bins)

    if rel_coordinates:
        x, y = x + pitch_dimensions[0] / 2, y + pitch_dimensions[1] / 2

    # Extract bin values [0, num bins - 1]
    bin_x = int(min(np.floor(x / bin_x_width), num_x_bins - 1))
    bin_y = int(min(np.floor(y / bin_y_width), num_y_bins - 1))

    if output == 'bin_ix':
        return bin_x, bin_y
    elif output == 'bin_rel':
        return f"({str(bin_x + 1)}/{str(num_x_bins)}, {str(bin_y + 1)}/{str(num_y_bins)})"
    else:
        x_labels, y_labels = bin_names['x'][num_x_bins], bin_names['y'][num_y_bins]
        return x_labels[bin_x], y_labels[bin_y]
