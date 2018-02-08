# %matplotlib inline

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier


# data directory
DATA_DIR = os.path.join('../..', 'pover-t', 'data')

# data_paths = {'A': {'train': os.path.join(DATA_DIR, 'A', 'A_hhold_train.csv'), 
#                     'test':  os.path.join(DATA_DIR, 'A', 'A_hhold_test.csv')}, 
#               
#               'B': {'train': os.path.join(DATA_DIR, 'B', 'B_hhold_train.csv'), 
#                     'test':  os.path.join(DATA_DIR, 'B', 'B_hhold_test.csv')}, 
#               
#               'C': {'train': os.path.join(DATA_DIR, 'C', 'C_hhold_train.csv'), 
#                     'test':  os.path.join(DATA_DIR, 'C', 'C_hhold_test.csv')}}

data_paths = {'A': {'train': os.path.join(DATA_DIR, 'train', 'A_hhold_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'test', 'A_hhold_test.csv')}, 
              
              'B': {'train': os.path.join(DATA_DIR, 'train', 'B_hhold_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'test', 'B_hhold_test.csv')}, 
              
              'C': {'train': os.path.join(DATA_DIR, 'train', 'C_hhold_train.csv'), 
                    'test':  os.path.join(DATA_DIR, 'test', 'C_hhold_test.csv')}}
# load training data
a_train = pd.read_csv(data_paths['A']['train'], index_col='id')
b_train = pd.read_csv(data_paths['B']['train'], index_col='id')
c_train = pd.read_csv(data_paths['C']['train'], index_col='id')

print(type(a_train))
print(a_train)


a_train_arr = a_train[['maLAYXwi']].as_matrix()
print(a_train_arr)
