import numpy as np
import pandas as pd

from TaSS import load_data
from TaSS import preprocess
from TaSS import learn
from TaSS import submit

# Read household and individual training and test data
# for each country
hh_train, hh_test = load_data.read_household_data()
indiv_train, indiv_test = load_data.read_individual_data()


#Pre-process household train and test data for each country
print("Country A")
aX_train = preprocess.pre_process_data(hh_train[0].drop('poor', axis=1))
ay_train = np.ravel(hh_train[0].poor)
aX_test = preprocess.pre_process_data(hh_test[0], enforce_cols=aX_train.columns)
print()

print("Country B")
bX_train = preprocess.pre_process_data(hh_train[1].drop('poor', axis=1))
by_train = np.ravel(hh_train[1].poor)
bX_test = preprocess.pre_process_data(hh_test[1], enforce_cols=bX_train.columns)
print()

print("Country C")
cX_train = preprocess.pre_process_data(hh_train[2].drop('poor', axis=1))
cy_train = np.ravel(hh_train[2].poor)
cX_test = preprocess.pre_process_data(hh_test[2], enforce_cols=cX_train.columns)
print()
print()

# Find optimal learning model for each country
#TODO: consider precision/recall in addition to accuracy in selecting optimal classifier
print("Classifiers for country A")
model_a = learn.classifier_comparison(aX_train, ay_train)
# model_a = learn.classifier_comparison(aX_train, ay_train, crossValidation=10)
print()

#TODO: missing values for country B, interpolation must not be working as expected
# print("Classifiers for country B")
# model_b = learn.classifier_comparison(bX_train, by_train)
# model_b = learn.classifier_comparison(bX_train, by_train, crossValidation=10)
print()

print("Classifiers for country C")
model_c = learn.classifier_comparison(cX_train, cy_train)
print(model_c)
# model_c = learn.classifier_comparison(cX_train, cy_train, crossValidation=10)
print()

#TODO: verify these lines actually work
# Get poor prediction values from optimal classifier of each country and output for competition
# submission = submit.convert_prediction_to_dataframe(model_a, aX_test, model_b, bX_test, model_c, cX_test)
# submission.to_csv('submission.csv')
