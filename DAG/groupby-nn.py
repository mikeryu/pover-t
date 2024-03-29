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
    #print("a_train_ind.shape")
    #print(a_train_ind.shape)
    print("a_train_hhold.shape")
    print(a_train_hhold.shape)
    print("b_train_hhold.shape")
    print(b_train_hhold.shape)
    print("c_train_hhold.shape")
    print(c_train_hhold.shape)
    print("a_train_ind.shape")
    print(a_train_ind.shape)
    print("b_train_ind.shape")
    print(b_train_ind.shape)
    print("c_train_ind.shape")
    print(c_train_ind.shape)

    # Check for duplicate column names between individual and household
    a_hhold_headers = list(a_train_hhold)
    b_hhold_headers = list(b_train_hhold)
    c_hhold_headers = list(c_train_hhold)
    a_ind_headers = list(a_train_ind)
    b_ind_headers = list(b_train_ind)
    c_ind_headers = list(c_train_ind)
    print("A Duplicate columns")
    print(set(a_hhold_headers).intersection(a_ind_headers))
    print("B Duplicate columns")
    print(set(b_hhold_headers).intersection(b_ind_headers))
    print("C Duplicate columns")
    print(set(c_hhold_headers).intersection(c_ind_headers))

    # Add columns of aggregated individual data to household data
    a_group = group_data(a_train_ind, a_train_hhold)
    print("a_group.shape")
    print(a_group.shape)
    print(a_group)

    # B individual data has lots of empty cells - need to clean it up
    b_group = remove_null_values_and_group_data(b_train_ind, b_train_hhold)
    print("b_group.shape")
    print(b_group.shape)

    c_group = group_data(c_train_ind, c_train_hhold)
    print("c_group.shape")
    print(c_group.shape)

    # Trim data that gives same answers for poor and non-poor
    #a_group_trim = trim_non_unique(a_group, 1)
    #print("a_group_trim.shape")
    #print(a_group_trim.shape)
    #b_group_trim = trim_non_unique(b_group, 1)
    #print("b_group_trim.shape")
    #print(b_group_trim.shape)
    #c_group_trim = trim_non_unique(c_group, 1)
    #print("c_group_trim.shape")
    #print(c_group_trim.shape)

    #return 0

    a_data = a_group
    b_data = b_group
    c_data = c_group

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

    # DON'T USE THIS - CREATES THE WRONG SIZE TEST DATA FOR SUBMISSION
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
    print(c_sub)

    # combine predictions and save for submission
    submission = pd.concat([a_sub, b_sub, c_sub])

    print("Submission head:")
    print(submission.head())
    print("\nSubmission tail:")
    print(submission.tail())

    print("Converting to csv for submission...")
    submission.to_csv('groupby_submission_4.csv')
    print("All done")


# group_data takes in individual data
# Creates a column of data aggregated by household id that consists of the most frequent
# individual response
# Example:
# id      iid      cKfjueDD
# 1       1        WkxyUi
# 1       2        MhgpsA
# 1       3        WkxyUi
# Becomes
# id      cKfjueDD
# 1       WkxyUi     ------> where WkxyUi is the most frequent response from individuals in household 1
# This new column is added to the household data by household id
def group_data(idf, hdf):
    print("\n ==== Group By ====")
    # Get the column headers from the dataframe of individual data
    headers = list(idf)
    headers.remove('iid')
    # Remove duplicate columns between individual and household
    headers.remove('country')
    headers.remove('poor')
    # Special case of duplicate column in country B
    if 'wJthinfa' in headers:
        headers.remove('wJthinfa')

    hhold_group = hdf

    # Need to create new dataframe column of most frequent responses grouped by household id
    for h in headers:
        # Create a column that consists of the household id and the most frequent response value
        new_column = idf.groupby('id')[[h]].agg(lambda x:x.value_counts().index[0])
        # Add that column to the household data
        hhold_group = pd.concat([hhold_group, new_column], axis=1, join_axes=[hhold_group.index])

    # Add a column that consists of the count of individuals in a household
    # Not sure how to add a column header (GkLMwxSq)
    ind_count = idf.groupby('id')[['iid']].count()
    hhold_group = pd.concat([hhold_group, ind_count], axis=1, join_axes=[hhold_group.index])

    return hhold_group

# remove_null_values_and_group_data handles the data that contains empty cells
# The agg function in groupby does not work if there are no values (size 0)
# This function separates the data into columns that have null values and columns that do not
# The null values are removed before running aggregation and then those columns are concatenated to the original dataframe
# The remaining columns are also aggregated and concatenated
def remove_null_values_and_group_data(idf, hdf):
    print("\n===========  REMOVE NULLS ==============")
    # Get columns with empty cells (NaN values in dataframe)
    cols_to_drop = idf.columns[idf.isna().any()].tolist()
    hhold_group = hdf

    for col in cols_to_drop:
        # Remove rows with null values
        idf_not_null = idf[[col]].dropna()
        # Create a column that consists of the household id and the most frequent response value
        new_column = idf_not_null.groupby('id')[[col]].agg(lambda x:x.value_counts().index[0])
        # Add that column to the household data
        hhold_group = pd.concat([hhold_group, new_column], axis=1, join_axes=[hhold_group.index])

    #Create a new dataframe of the columns with NaN values
    #nan_df = idf.loc[:, idf.isna().any()]

    # Process the rest of the original dataframe
    idf_not_nan = idf.drop(cols_to_drop, axis = 1)
    print("idf_not_nan.shape")
    print(idf_not_nan.shape)

    return group_data(idf_not_nan, hhold_group)


# Drop columns in data that have little to no variation (same answers for poor and non-poor)
def trim_non_unique(df, max_nonuniques):
    print("\n ======== TRIM DATA =========\n")
    nonuniques = df.nunique()
    cols_to_drop = [col for col in nonuniques.index if nonuniques[col] <= max_nonuniques]
    # Need columans for poor and country
    cols_to_drop.remove('poor')
    cols_to_drop.remove('country')
    #print(cols_to_drop)

    df_trim = df.drop(cols_to_drop, axis=1)
    print(df_trim.shape)
    return df_trim


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

    print("\n=============================================\n")
    print("A Training")
    #print(a_train.head())
    print(a_train.info())
    print("\n=============================================\n")
    print("B Training")
    #print(b_train.head())
    print(b_train.info())
    print("\n=============================================\n")
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
