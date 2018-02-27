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
from keras import optimizers
import keras.backend as K

# data directory
DATA_DIR = os.path.join('../..', 'pover-t', 'data')
A_TRAIN_HHOLD = os.path.join(DATA_DIR, 'train', 'A_hhold_train.csv')
B_TRAIN_HHOLD = os.path.join(DATA_DIR, 'train', 'B_hhold_train.csv')
C_TRAIN_HHOLD = os.path.join(DATA_DIR, 'train', 'C_hhold_train.csv')
A_TRAIN_IND = os.path.join(DATA_DIR, 'train', 'A_indiv_train.csv')
B_TRAIN_IND = os.path.join(DATA_DIR, 'train', 'B_indiv_train.csv')
C_TRAIN_IND = os.path.join(DATA_DIR, 'train', 'C_indiv_train.csv')
A_TEST_HHOLD = os.path.join(DATA_DIR, 'test', 'A_hhold_test.csv')
B_TEST_HHOLD = os.path.join(DATA_DIR, 'test', 'B_hhold_test.csv')
C_TEST_HHOLD = os.path.join(DATA_DIR, 'test', 'C_hhold_test.csv')
A_TEST_IND = os.path.join(DATA_DIR, 'test', 'A_indiv_test.csv')
B_TEST_IND = os.path.join(DATA_DIR, 'test', 'B_indiv_test.csv')
C_TEST_IND = os.path.join(DATA_DIR, 'test', 'C_indiv_test.csv')

data_paths = {'A': {'train': A_TRAIN_HHOLD, 'test': A_TEST_HHOLD},
    'B': {'train': B_TRAIN_HHOLD, 'test': B_TEST_HHOLD},
    'C': {'train': C_TRAIN_HHOLD, 'test': C_TEST_HHOLD}}

ind_data_paths = {'A': {'train': A_TRAIN_IND, 'test': A_TEST_IND},
    'B': {'train': B_TRAIN_IND, 'test': B_TEST_IND},
    'C': {'train': C_TRAIN_IND, 'test': C_TEST_IND}}

def main():
    a_train_hhold, b_train_hhold, c_train_hhold, a_train_ind, b_train_ind,\
        c_train_ind = read_train_data()

    # Merge individual and household training data
    a_data = pd.concat([a_train_ind, a_train_hhold])
    b_data = pd.concat([b_train_ind, b_train_hhold])
    c_data = pd.concat([c_train_ind, c_train_hhold])

    # Process the training data
    print("Country A")
    a_train = preprocess_data(a_data.drop(columns=['poor']))
    a_labels = np.ravel(a_data.poor)

    print("\nCountry B")
    b_train = preprocess_data(b_data.drop(columns=['poor']))
    b_labels = np.ravel(b_data.poor)

    print("\nCountry C")
    c_train = preprocess_data(c_data.drop(columns=['poor']))
    c_labels = np.ravel(c_data.poor)

    print("\nTest Data")
    a_test_hhold, b_test_hhold, c_test_hhold, a_test_ind, b_test_ind,\
        c_test_ind = read_test_data()

    # Merge individual and household test data
    #a_test_data = pd.concat([a_test_ind, a_test_hhold])
    #b_test_data = pd.concat([b_test_ind, b_test_hhold])
    #c_test_data = pd.concat([c_test_ind, c_test_hhold])

    #Process the test data
    a_test = preprocess_data(a_test_hhold, enforce_cols=a_train.columns)
    b_test = preprocess_data(b_test_hhold, enforce_cols=b_train.columns)
    c_test = preprocess_data(c_test_hhold, enforce_cols=c_train.columns)

    # Train and predict over the data sets
    a_preds = train_and_predict(a_train, a_labels, a_test)
    a_sub = make_country_sub(a_preds, a_test, 'A')

    b_preds = train_and_predict(b_train, b_labels, b_test)
    b_sub = make_country_sub(b_preds, b_test, 'B')

    c_preds = train_and_predict(c_train, c_labels, c_test)
    c_sub = make_country_sub(c_preds, c_test, 'C')

    # combine predictions and save for submission
    submission = pd.concat([a_sub, b_sub, c_sub])

    print("Submission head:")
    print(submission.head())
    print("\nSubmission tail:")
    print(submission.tail())

    print("Converting to csv for submission...")
    submission.to_csv('merge_nn_submission_v2.csv')
    print("All done")

def train_and_predict(train, ids, test):
    model = Sequential()

    # Add an input layer
    model.add(Dense(72, activation='relu', input_shape=(train.shape[1],)))
    # Add some hidden layers
    model.add(Dense(36, activation='relu'))
    model.add(Dense(36, activation='relu'))
    model.add(Dense(36, activation='sigmoid'))
    model.add(Dense(36, activation='sigmoid'))
    # Add an output layer
    model.add(Dense(1, activation='sigmoid'))
    print("Model output shape:")
    model.output_shape
    print("Model summary:")
    model.summary()
    print("Model config:")
    model.get_config()
    print("Model weights:")
    model.get_weights()

    # Compile the model and fit the model to the data
    model.compile(loss='mean_squared_error',
                  optimizer='sgd',
                  metrics=['accuracy', precision, recall, fmeasure])

    model.fit(train, ids, epochs=20, batch_size=36, verbose=1)
    score = model.evaluate(train, ids, verbose=1)
    print(score)

    preds = model.predict(test)

    return preds

def read_train_data():
    # load training data
    a_train = pd.read_csv(data_paths['A']['train'], index_col='id')
    b_train = pd.read_csv(data_paths['B']['train'], index_col='id')
    c_train = pd.read_csv(data_paths['C']['train'], index_col='id')
    a_indiv_train = pd.read_csv(ind_data_paths['A']['train'], index_col='id')
    b_indiv_train = pd.read_csv(ind_data_paths['B']['train'], index_col='id')
    c_indiv_train = pd.read_csv(ind_data_paths['C']['train'], index_col='id')

    print("\n\n=============================================\n")
    print("A Training")
    #print(a_train.head())
    print(a_train.info())
    print("\n\n=============================================\n")
    print("B Training")
    #print(b_train.head())
    print(b_train.info())
    print("\n\n=============================================\n")
    print("C Training")
    #print(c_train.head())
    print(c_train.info())

    return a_train, b_train, c_train, a_indiv_train, b_indiv_train,\
        c_indiv_train

def read_test_data():
    # load test data
    a_test = pd.read_csv(data_paths['A']['test'], index_col='id')
    b_test = pd.read_csv(data_paths['B']['test'], index_col='id')
    c_test = pd.read_csv(data_paths['C']['test'], index_col='id')
    a_indiv_test = pd.read_csv(ind_data_paths['A']['test'], index_col='id')
    b_indiv_test = pd.read_csv(ind_data_paths['B']['test'], index_col='id')
    c_indiv_test = pd.read_csv(ind_data_paths['C']['test'], index_col='id')

    return a_test, b_test, c_test, a_indiv_test, b_indiv_test, c_indiv_test

# Standardize features
def standardize(df, numeric_only=True):
    numeric = df.select_dtypes(include=['int64', 'float64'])

    # subtract mean and divide by std
    df[numeric.columns] = (numeric - numeric.mean()) / numeric.std()

    return df

def preprocess_data(df, enforce_cols=None):
    print("Input shape:\t{}".format(df.shape))

    df = standardize(df)
    print("After standardization {}".format(df.shape))

    # create dummy variables for categoricals
    df = pd.get_dummies(df)
    print("After converting categoricals:\t{}".format(df.shape))


    # match test set and training set columns
    if enforce_cols is not None:
        to_drop = np.setdiff1d(df.columns, enforce_cols)
        to_add = np.setdiff1d(enforce_cols, df.columns)

        df.drop(to_drop, axis=1, inplace=True)
        df = df.assign(**{c: 0 for c in to_add})

    df.fillna(0, inplace=True)

    return df

# save submission
def make_country_sub(preds, test_feat, country):
    # make sure we code the country correctly
    country_codes = ['A', 'B', 'C']

    # get just the poor probabilities
    country_sub = pd.DataFrame(data=preds[:, 0],  # proba p=1
                               columns=['poor'],
                               index=test_feat.index)


    # add the country code for joining later
    country_sub["country"] = country
    return country_sub[["country", "poor"]]


# From previous keras version
# https://github.com/keras-team/keras/commit/a56b1a55182acf061b1eb2e2c86b48193a0e88f7 
def precision(y_true, y_pred):
    """Precision metric.
 
    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

# From previous keras version
# https://github.com/keras-team/keras/commit/a56b1a55182acf061b1eb2e2c86b48193a0e88f7 
def recall(y_true, y_pred):
    """Recall metric.
 
    Only computes a batch-wise average of recall.
 
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

# From previous keras version
# https://github.com/keras-team/keras/commit/a56b1a55182acf061b1eb2e2c86b48193a0e88f7 
def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.
 
    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.

    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.

    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score


# From previous keras version
# https://github.com/keras-team/keras/commit/a56b1a55182acf061b1eb2e2c86b48193a0e88f7 
def fmeasure(y_true, y_pred):
    """Computes the f-measure, the harmonic mean of precision and recall.
 
    Here it is only computed as a batch-wise average, not globally.
    """
    return fbeta_score(y_true, y_pred, beta=1) 


if __name__ == '__main__':
    main()
