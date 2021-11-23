"""
Data processing module of football2vec. Contains classes: FootballTokenizer, Corpus.
Also contains the build_data_objects function and its nested function for building the core data objects.
"""

import numpy as np
import pandas as pd
import ast
from tqdm import tqdm
import os
import pickle
import plotly.express as px
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer

from player2vec.data_handler import load_all_events_data
from player2vec.params import COLUMNS
from player2vec.utils import get_location_bin


class FootballTokenizer:
    def __init__(self, **kwargs):
        self.tokens_encoder = OrdinalEncoder()
        self.num_x_bins = kwargs.get('num_x_bins', 5)
        self.num_y_bins = kwargs.get('num_y_bins', 5)
        self.actions_to_ignore_outcome = kwargs.get('actions_to_ignore_outcome', ['duel'])

    def tokenize_action(self, action: pd.Series) -> str:
        '''
        Convert action - a record of StatsBomb events data - to a string token
        :param action: Series, a single action
        :return: token - string
        '''
        action_name = action.get('type', action.get(COLUMNS.ACTION_TYPE, np.nan))
        if action_name is np.nan:
            return np.nan
        else:
            action_name = action_name.lower()

        token = f'<{action_name}>'

        # Add location
        if action['location'] is not np.nan and action['location'] is not np.nan:
            if isinstance(action['location'], str):
                x, y = ast.literal_eval(action['location'])
            elif isinstance(action['location'], list) or isinstance(action['location'], tuple):
                x, y = action['location']
            else:
                raise ValueError("Unfamiliar value for action location:", action['location'])
            location_bin = get_location_bin(x, y, num_x_bins=self.num_x_bins, num_y_bins=self.num_y_bins)
            token = f"{location_bin}".replace(" ", "") + token

        return token.replace(" ", "_")


class Corpus:
    def __init__(self, **kwargs):
        '''
        corpus - list of paragraphs. Resolution/ aggregation is determined by self.aggr_columns.
            self.aggr_columns aggregation is executed by string grouping, separated by self.separator
            Tokens of language are stored in vocabulary, and their encodings in vocabulary_ix.
            Transformation from tokens <-> tokens encodings are allowed by self.ix_2_token & self.token_2_ix.
        documents - name of documents, when used for Doc2Vec
        :param kwargs:
        :type kwargs:
        '''
        self.aggr_columns = kwargs.get('aggr_columns', None)
        if self.aggr_columns is None:
            self.aggr_columns = ['match_id', 'period', 'possession']
        self.ft = kwargs.get('tokenizer', FootballTokenizer(
            actions_to_ignore_outcome=kwargs.get('actions_to_ignore_outcome', ['duel'])))

        self.separator = kwargs.get('separator', '-')
        # Init None attributes
        self.corpus = None
        self.vocabulary = None
        self.vocabulary_ix = None
        self.ix_2_token = None
        self.token_2_ix = None
        self.documents_names = None
        self.verbose = kwargs.get('verbose', False)

    def build_corpus(self, events_data: pd.DataFrame, allow_concat_documents_=True, **kwargs) -> pd.DataFrame:
        '''
        Build corpus using given vocab_data. Associates actions with matching tokens, aggregate them to sentences,
                    and then to documents.
        :param events_data: pd.DataFrame of StatsBomb events data
        :param allow_concat_documents_: whether to allow concatenation of sentences to documents if < min length limit
        :param kwargs:
        :return: vocab_data with new 'token' column. All object attributes are updated.
        '''
        if self.verbose:
            print(f"vocab_data size: {events_data.shape}\n")

        vocab_data = events_data.copy()

        if self.verbose: print('\nStart Tokenization')
        vocab_data['token'] = vocab_data.progress_apply(lambda action:
                                                        self.ft.tokenize_action(action),
                                                        axis=1)
        events_data['token'] = vocab_data['token'].copy()
        vocab_data = vocab_data[~vocab_data['token'].isna()]
        if self.verbose:
            print('Done.')
            print(f"Vocab_data size after processing and removing NAs tokens: {events_data.shape}\n")

        vocabulary = [val for val in vocab_data['token'].unique() if val is not np.nan]
        vocabulary.extend(['oov'])
        if self.verbose:
            print(f'Raw length of vocabulary: (including oov)', len(vocabulary))

        # Create mappers of token to index and vice versa
        ix_2_token = dict(enumerate(vocabulary))
        ix_2_token = {str(key): val for key, val in ix_2_token.items()}
        token_2_ix = {val: key for key, val in ix_2_token.items()}

        # Set the appropriate token index for each action
        vocab_data['token_ix'] = vocab_data['token'].apply(
            lambda token: token_2_ix.get(token, token_2_ix['oov']))

        # Keep only columns relevant for sentences grouping
        vocab_data = vocab_data[['token_ix', 'token'] + self.aggr_columns]

        for col in self.aggr_columns:
            vocab_data[col] = vocab_data[col].astype(str)
        vocab_data['aggr_key'] = vocab_data[self.aggr_columns].apply(
            lambda vec: self.separator.join(vec), axis=1)

        # Create sentences and documents
        sentences = vocab_data[['aggr_key', 'token_ix']].groupby('aggr_key')
        sentences = sentences['token_ix'].agg(list).reset_index()
        documents = sentences['aggr_key'].tolist()
        sentences = sentences['token_ix'].tolist()

        sampling_window = kwargs.get('sampling_window', 5)
        corpus = []
        if self.verbose:
            print('\nBuilding sentences...')

        if not allow_concat_documents_:
            # If we can't concatenate sentences --> add to corpus sentences that are longer than min threshold
            self.documents_names = []
            if self.verbose: print('\nBuilding Documents...')
            for i, doc_ in tqdm(enumerate(sentences)):
                if len(doc_) >= sampling_window:
                    corpus.append(doc_[:])
                    self.documents_names.append(documents[i])
            if self.verbose: print('Final number of documents_names:', len(self.documents_names))
        else:
            # Paragraphs can be merged and concatenated
            # If we can concatenate multiple short sentences (shorter than min threshold) to longer sentences > merge
            if self.verbose:
                print('\nConcatenating Documents to build sampling_window sized documents...')
            cum_actions_length = 0
            cum_actions = []

            for sentence_ in tqdm(sentences):
                if len(sentence_) >= sampling_window:
                    corpus.append(sentence_[:])
                else:
                    cum_actions.extend(sentence_[:])
                    cum_actions_length += len(sentence_)

                    if cum_actions_length >= sampling_window:
                        corpus.append(cum_actions[:])
                        cum_actions_length = 0
                        cum_actions = []

        if self.verbose:
            print('Final number of sentences:', len(corpus))

        # Update vocabulary
        corpus_flat = set([subitem for item in corpus for subitem in item if type(item) is list])
        vocaulary_ix = set([token_2_ix[token_] for token_ in vocabulary])

        # Update vocabulary after merging sentences
        vocabulary_ix = corpus_flat.intersection(vocaulary_ix)
        vocabulary = [ix_2_token[token_ix] for token_ix in vocabulary_ix]
        if self.verbose:
            print('Final length of vocabulary:', len(vocabulary))

        # Set class properties
        self.corpus = corpus
        self.vocabulary = vocabulary
        self.vocabulary_ix = vocabulary_ix
        self.ix_2_token = ix_2_token
        self.token_2_ix = token_2_ix

        return events_data
