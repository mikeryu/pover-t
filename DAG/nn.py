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
a_train = pd.read_csv(data_paths['A']['train'], index_col='id')
b_train = pd.read_csv(data_paths['B']['train'], index_col='id')
c_train = pd.read_csv(data_paths['C']['train'], index_col='id')
a_indiv_train = pd.read_csv(ind_data_paths['A']['train'], index_col='id')
b_indiv_train = pd.read_csv(ind_data_paths['B']['train'], index_col='id')
c_indiv_train = pd.read_csv(ind_data_paths['C']['train'], index_col='id')

print("\n\n=============================================\n\n")
print("A Training")
print(a_train.head())
#b_train.head()
print("\n\n=============================================\n\n")
print("C Training")
print(c_train.head())

#a_train.poor.value_counts().plot.bar(title='Number of Poor for country A')
#plt.show()
#b_train.poor.value_counts().plot.bar(title='Number of Poor for country B')
#plt.show()
#c_train.poor.value_counts().plot.bar(title='Number of Poor for country C')
#plt.show()


print("\n\n=============================================\n\n")
print(a_train.info())
print(b_train.info())
print(c_train.info())

# Standardize features
def standardize(df, numeric_only=True):
    numeric = df.select_dtypes(include=['int64', 'float64'])

    # subtracy mean and divide by std
    df[numeric.columns] = (numeric - numeric.mean()) / numeric.std()

    return df


def pre_process_data(df, enforce_cols=None):
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


print("Country A")
aX_train = pre_process_data(a_train.drop('poor', axis=1))
ay_train_temp = np.ravel(a_train.poor)
print(aX_train.head())

ay_train_temp.flatten()

ay_train_list = []

for item in range(0, 8203):
    if (ay_train_temp[item] == False):
        ay_train_list.append(0)
    else:
        ay_train_list.append(1)

ay_train = np.asarray(ay_train_list)

#print(ay_train.tolist())

print("\nCountry B")
bX_train = pre_process_data(b_train.drop('poor', axis=1))
by_train = np.ravel(b_train.poor)

print("\nCountry C")
cX_train = pre_process_data(c_train.drop('poor', axis=1))
cy_train = np.ravel(c_train.poor)


def train_model(features, labels, **kwargs):    
    # instantiate model
    model = RandomForestClassifier(n_estimators=50, random_state=0)
    
    # train model
    model.fit(features, labels)
    
    # get a (not-very-useful) sense of performance
    accuracy = model.score(features, labels)
    print(f"In-sample accuracy: {accuracy:0.2%}")
    
    return model



print("Train model a")
model_a = train_model(aX_train, ay_train)
print("Train model b")
model_b = train_model(bX_train, by_train)
print("Train model c")
model_c = train_model(cX_train, cy_train)
print("\n\n=============================================\n\n")


# load test data
a_test = pd.read_csv(data_paths['A']['test'], index_col='id')
b_test = pd.read_csv(data_paths['B']['test'], index_col='id')
c_test = pd.read_csv(data_paths['C']['test'], index_col='id')

# process the test data
a_test = pre_process_data(a_test, enforce_cols=aX_train.columns)
b_test = pre_process_data(b_test, enforce_cols=bX_train.columns)
c_test = pre_process_data(c_test, enforce_cols=cX_train.columns)
print("\n\n=============================================\n\n")


# make predictions
a_preds = model_a.predict_proba(a_test)
b_preds = model_b.predict_proba(b_test)
c_preds = model_c.predict_proba(c_test)

# save submission
def make_country_sub(preds, test_feat, country):
    # make sure we code the country correctly
    country_codes = ['A', 'B', 'C']

    # get just the poor probabilities
    country_sub = pd.DataFrame(data=preds[:, 1],  # proba p=1
                               columns=['poor'],
                               index=test_feat.index)


    # add the country code for joining later
    country_sub["country"] = country
    return country_sub[["country", "poor"]]

# convert preds to data frames
a_sub = make_country_sub(a_preds, a_test, 'A')
b_sub = make_country_sub(b_preds, b_test, 'B')
c_sub = make_country_sub(c_preds, c_test, 'C')


# combine predictions and save for submission
submission = pd.concat([a_sub, b_sub, c_sub])

print("Submission head:")
print(submission.head())
print("\nSubmission tail:")
print(submission.tail())

#print("Converting to csv for submission...")
#submission.to_csv('nn_submission.csv')
#print("All done")


print("\n\n=================== KERAS  ==========================\n\n")

# Initialize the constructor
model = Sequential()
#modelInput = Input(shape=(8023, 1))
#modelDense = Dense(8023)(modelInput)
#model = Model(inputs=modelInput, outputs=modelDense)

# Add an input layer
model.add(Dense(12, activation='relu', input_shape=(859,)))
# Add one hidden layer
model.add(Dense(8, activation='relu'))
# Add an output layer
model.add(Dense(1, activation='softmax'))

model.output_shape
model.summary()
model.get_config()
model.get_weights()

# Compile the model and fit the model to the data
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr=0.001),
              metrics=['accuracy'])

model.fit(aX_train, ay_train, epochs=1, batch_size=1, verbose=1)
score = model.evaluate(aX_train, ay_train, verbose=1)
print(score)

a_pred = model.predict_classes(a_test)
print(a_pred.tolist())
