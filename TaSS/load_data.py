# %matplotlib inline

import os

import numpy as np
import pandas as pd
import sklearn as sk

import matplotlib.pyplot as plt


class Loader:

    def read_data(self):
        # data directory
        DATA_DIR = os.path.join('..', 'data', 'processed')

        data_paths = {'A': {'train': os.path.join(DATA_DIR, 'A', 'A_hhold_train.csv'),
                            'test': os.path.join(DATA_DIR, 'A', 'A_hhold_test.csv')},

                      'B': {'train': os.path.join(DATA_DIR, 'B', 'B_hhold_train.csv'),
                            'test': os.path.join(DATA_DIR, 'B', 'B_hhold_test.csv')},

                      'C': {'train': os.path.join(DATA_DIR, 'C', 'C_hhold_train.csv'),
                            'test': os.path.join(DATA_DIR, 'C', 'C_hhold_test.csv')}}


        # load training data
        a_train = pd.read_csv(data_paths['A']['train'], index_col='id')
        b_train = pd.read_csv(data_paths['B']['train'], index_col='id')
        c_train = pd.read_csv(data_paths['C']['train'], index_col='id')

        return (a_train, b_train, c_train)

#preprocessing class
    def preprocess(self):
        pass
