# %matplotlib inline

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential, Model
from keras.layers import Dense, Input


# data directory
DATA_DIR = os.path.join('../..', 'pover-t', 'data')

data_paths = {'A': {'train': os.path.join(DATA_DIR, 'train', 'A_hhold_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'test', 'A_hhold_test.csv')}, 
              
              'B': {'train': os.path.join(DATA_DIR, 'train', 'B_hhold_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'test', 'B_hhold_test.csv')}, 
              
              'C': {'train': os.path.join(DATA_DIR, 'train', 'C_hhold_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'test', 'C_hhold_test.csv')}}

ind_data_paths = {'A': {'train': os.path.join(DATA_DIR, 'train', 'A_indiv_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'test', 'A_indiv_test.csv')}, 
              
              'B': {'train': os.path.join(DATA_DIR, 'train', 'B_indiv_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'test', 'B_indiv_test.csv')}, 
              
              'C': {'train': os.path.join(DATA_DIR, 'train', 'C_indiv_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'test', 'C_indiv_test.csv')}}

# load training data
#a_train = pd.read_csv(data_paths['A']['train'], index_col='id')
#b_train = pd.read_csv(data_paths['B']['train'], index_col='id')
#c_train = pd.read_csv(data_paths['C']['train'], index_col='id')
#a_indiv_train = pd.read_csv(ind_data_paths['A']['train'], index_col='id')
#b_indiv_train = pd.read_csv(ind_data_paths['B']['train'], index_col='id')
#c_indiv_train = pd.read_csv(ind_data_paths['C']['train'], index_col='id')

# load testing data
a_test = pd.read_csv(data_paths['A']['test'], index_col='id')
b_test = pd.read_csv(data_paths['B']['test'], index_col='id')
c_test = pd.read_csv(data_paths['C']['test'], index_col='id')
a_indiv_test = pd.read_csv(ind_data_paths['A']['test'], index_col='id')
b_indiv_test = pd.read_csv(ind_data_paths['B']['test'], index_col='id')
c_indiv_test = pd.read_csv(ind_data_paths['C']['test'], index_col='id')


# Convert test data from text to integers for use in Keras model
# Store as csv
rowNum = 0
colNum = 0
for column in c_indiv_test:
    rowNum = 0
    for row in c_indiv_test[column]:
        c_indiv_test.iloc[rowNum,colNum] = abs(hash(row)) % (10 ** 8)
        rowNum += 1
    colNum += 1

c_indiv_test.to_csv("data/test/C_indiv_test_mod.csv")


