import numpy as np
import pandas as pd

from TaSS import load_data
from TaSS import preprocess
from TaSS import learn
from TaSS import submit


def main():
    # Read household and individual training and test data for each country
    hh_train, hh_test = load_data.read_household_data()
    indiv_train, indiv_test = load_data.read_individual_data()

    # Just labels for printing so we know which is which
    labels = ["Country A", "Country B", "Country C"]

    # Iterate through each country and preprocess + train
    models = preprocess_and_run_classification(labels, hh_test, hh_train)

    # Get poor prediction values from optimal classifier of each country and output for competition
    # TODO: verify these lines actually work
    submission = submit.convert_prediction_to_dataframe(*models, *hh_test)
    submission.to_csv('submission.csv')


def preprocess_and_run_classification(labels, hh_test, hh_train):
    # for each country
    models = []
    for i in range(len(hh_test)):
        # preprocess the household test and train data
        # TODO: missing values for country B, interpolation must not be working as expected
        print(labels[i], "*" * 50)
        X_train, y_train, X_test = preprocess_household_test_train_data(hh_train[i], hh_test[i])
        print( "*" * 50, "DONE", end='\n\n')

        # train different classifiers and report on the result
        # TODO: consider precision/recall in addition to accuracy in selecting optimal classifier
        print("Classifier for", labels[i],  "*" * 50)
        model = learn.classifier_comparison(X_train, y_train)
        models.append(model)
        print(model, end='\n\n')

    return tuple(models)

def preprocess_household_test_train_data(hh_train, hh_test):
    X_train = preprocess.pre_process_data(hh_train.drop('poor', axis=1))
    y_train = np.ravel(hh_train.poor)
    X_test = preprocess.pre_process_data(hh_test, enforce_cols=X_train.columns)
    return X_train, y_train, X_test


if __name__ == '__main__':
    main()
