"""
This module holds definitions for identifying patterns, actions or a collection of them by type, outcome, etc.
It is mainly used by explaines.py and data_processing.py. Some are not used and are merely for further demonstration.
"""


import ast
import re
import warnings
import numpy as np
import pandas as pd

from player2vec.params import COLUMNS, CONSTANTS


class Pattern(object):
    def __init__(self, words_to_match: list, words_to_exclude: list):
        self.words_to_match = words_to_match
        self.words_to_exclude = words_to_exclude

    def search(self, _str: str):
        raise NotImplementedError("")


class ANDPattern(Pattern):
    def search(self, _str: str):
        for _word in self.words_to_exclude:
            if _word in _str:
                return False
        for _word in self.words_to_match:
            if _word not in _str:
                return False
        return True


class ORPattern(Pattern):
    def search(self, _str: str):
        for _word in self.words_to_exclude:
            if _word in _str:
                return False
        for _word in self.words_to_match:
            if _word in _str:
                return True
        return False


# Regex patterns of tokens families
in_box_scoring_pattern = re.compile("\(5\/5,[2-4]{1}\/5\)\<shot\>")
out_box_scoring_pattern = re.compile("\([3-4]\/5,[1-5]{1}\/5\)\<shot\>")

passes_to_right_pattern = re.compile("\(3\/5,3\/5\)\<pass\>\:\( \- \>")
passes_forward_pattern = re.compile("\(3\/5,3\/5\)\<pass\>\:\( \^ \|")
passes_from_right_flank_pattern = re.compile("\(5\/5,5\/5\)\<pass\>\:\( \- \<")

forward_pressure_pattern = re.compile("\([3-5]\/5,[1-5]{1}\/5\)\<pressure\>")

dribble_pattern = re.compile("\([3-5]\/5,[1-5]{1}\/5\)\<dribble\>:\|outcome=complete")
dribble_past_pattern = re.compile("\([3-5]\/5,[1-5]{1}\/5\)\<dribbled_past\>")
flank_dribble_pattern = re.compile("\([1,5]\/5,[1-5]{1}\/5\)\<dribble\>:\|outcome=complete")
flank_dribble_past_pattern = re.compile("\(1,5]\/5,[1-5]{1}\/5\)\<dribbled_past\>")

# Improving
better_shots = [{'pattern': ANDPattern(['<shot>', 'outcome=blocked'], []),
                 'switch_from': 'outcome=blocked',
                 'switch_to': 'outcome=goal'},
                {'pattern': ANDPattern(['<shot>', 'outcome=wayward'], []),
                 'switch_from': 'outcome=wayward',
                 'switch_to': 'outcome=goal'},
                {'pattern': ANDPattern(['<shot>', 'outcome=saved'], []),
                 'switch_from': 'outcome=saved',
                 'switch_to': 'outcome=goal'},
                {'pattern': ANDPattern(['<shot>', 'outcome=off_t'], []),
                 'switch_from': 'outcome=off_t',
                 'switch_to': 'outcome=goal'},
                ]

passes_backwards = [{'pattern': ANDPattern(['<pass>', '^'], []),
                     'switch_from': '^',
                     'switch_to': 'v'}]

worse_shots = [{'pattern': ANDPattern(['<shot>', 'outcome=goal'], []),
                'switch_from': 'outcome=goal',
                'switch_to': 'outcome=wayward'}]

better_dribble = [{'pattern': ANDPattern(['<dribble>', 'outcome=incomplete'], []),
                   'switch_from': 'incomplete',
                   'switch_to': 'complete'}]

worse_dribble = [{'pattern': ANDPattern(['<dribble>', 'outcome=complete'], []),
                  'switch_from': 'complete',
                  'switch_to': 'incomplete'}]
switch_to_right_leg = [{'pattern': ORPattern(['left_foot'], []),
                        'switch_from': 'left_foot',
                        'switch_to': 'right_foot'}]


def _search(_pattern, _token):
    '''
    Search within _token using _pattern object - regex of Pattern
    :param _pattern: pattern to match in the token
    :type _pattern: regex pattern or Pattern object
    :param _token: token (str) to search in
    :return: regex search results / book result of Pattern
    '''
    if isinstance(_pattern, Pattern):
        return _pattern.search(_token)
    else:
        return re.search(_pattern, _token)


def get_tokens_by_regex_pattern(vocabulary: list, re_pattern):
    '''
        Function receives a vocabulary (list) and a regex pattern and return all matching tokens (or None if no match)
        :param vocabulary: list of string tokens that form our vocabulary
        :param re_pattern: regex pattern to match
        :return: Bool of the condition result
    '''
    relevant_tokens = [token for token in vocabulary if re.search(re_pattern, token)]
    if len(relevant_tokens) > 0:
        return relevant_tokens
    else:
        warnings.warn(f"No match for pattern {str(re_pattern)}")
        return []
