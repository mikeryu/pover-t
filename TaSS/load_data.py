# %matplotlib inline

import os
import pandas as pd



class Loader:

    @staticmethod
    def read_household_data():
        # data directory
        DATA_DIR = os.path.join('..', 'data')

        data_paths = {'A': {'train': os.path.join(DATA_DIR, 'train', 'A_hhold_train.csv'),
                            'test': os.path.join(DATA_DIR, 'test', 'A_hhold_test.csv')},

                      'B': {'train': os.path.join(DATA_DIR, 'train', 'B_hhold_train.csv'),
                            'test': os.path.join(DATA_DIR, 'test', 'B_hhold_test.csv')},

                      'C': {'train': os.path.join(DATA_DIR, 'train', 'C_hhold_train.csv'),
                            'test': os.path.join(DATA_DIR, 'test', 'C_hhold_test.csv')}}

        # load training data
        a_train = pd.read_csv(data_paths['A']['train'], index_col='id')
        b_train = pd.read_csv(data_paths['B']['train'], index_col='id')
        c_train = pd.read_csv(data_paths['C']['train'], index_col='id')
        a_test = pd.read_csv(data_paths['A']['test'], index_col='id')
        b_test = pd.read_csv(data_paths['B']['test'], index_col='id')
        c_test = pd.read_csv(data_paths['C']['test'], index_col='id')

        return ((a_train, b_train, c_train), (a_test, b_test, c_test))

    @staticmethod
    def read_individual_data():
        # data directory
        DATA_DIR = os.path.join('..', 'data')

        data_paths = {'A': {'train': os.path.join(DATA_DIR, 'train', 'A_indiv_train.csv'),
                            'test': os.path.join(DATA_DIR, 'test', 'A_indiv_test.csv')},

                      'B': {'train': os.path.join(DATA_DIR, 'train', 'B_indiv_train.csv'),
                            'test': os.path.join(DATA_DIR, 'test', 'B_indiv_test.csv')},

                      'C': {'train': os.path.join(DATA_DIR, 'train', 'C_indiv_train.csv'),
                            'test': os.path.join(DATA_DIR, 'test', 'C_indiv_test.csv')}}

        # load training data
        a_train = pd.read_csv(data_paths['A']['train'], index_col='id')
        b_train = pd.read_csv(data_paths['B']['train'], index_col='id')
        c_train = pd.read_csv(data_paths['C']['train'], index_col='id')
        a_test = pd.read_csv(data_paths['A']['test'], index_col='id')
        b_test = pd.read_csv(data_paths['B']['test'], index_col='id')
        c_test = pd.read_csv(data_paths['C']['test'], index_col='id')

        return ((a_train, b_train, c_train), (a_test, b_test, c_test))



