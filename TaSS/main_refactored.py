import numpy as np
import pickle
import sys
import os

from TaSS import load_data
from TaSS import preprocess
from TaSS import learn
from TaSS import submit


def main(args):
    # Read household and individual training and test data for each country
    hh_train, hh_test = load_data.read_household_data()
    indiv_train, indiv_test = load_data.read_individual_data()

    # Just labels for printing so we know which is which
    labels = ["Country A", "Country B", "Country C"]

    # Iterate through each country and preprocess + train
    models = preprocess_and_run_classification(labels, hh_test, hh_train,
                                               cv=int(args[0]) if args else None)

    # Get poor prediction values from optimal classifier of each country and output for competition
    # TODO: verify these lines actually work -- they do not :(
    # submission = submit.convert_prediction_to_dataframe(*models, *hh_test)
    # submission.to_csv('submission.csv')


def preprocess_and_run_classification(labels, hh_test, hh_train, cv):
    # for each country
    models = []
    for i in range(len(hh_test)):
        # preprocess the household test and train data
        # TODO: missing values for country B, interpolation must not be working as expected
        print(labels[i], '(Preprocessing)', "*" * 50)
        loaded = load_preprocessed_pickle(labels[i])

        if all(list(map(lambda x: x is not None, loaded))):
            X_train, y_train, X_test = loaded
            print('All preprocessed data loaded from previous pickle.')
        else:
            X_train, y_train, X_test = preprocess_household_test_train_data(hh_train[i], hh_test[i], labels[i])

        print("*" * 70, "DONE!", end='\n\n')

        print("dimensionality reduction")
        X_train = learn.reduce_dimensionality(X_train)

        # train different classifiers and report on the result
        # TODO: consider precision/recall in addition to accuracy in selecting optimal classifier
        print("Classifier for", labels[i], "*" * 50)
        model = learn.classifier_comparison(X_train, y_train, cv)
        models.append(model)
        print(model, end='\n\n')

    return tuple(models)


def preprocess_household_test_train_data(hh_train, hh_test, label):
    X_train = preprocess.pre_process_data(hh_train.drop('poor', axis=1))
    # print(X_train)
    y_train = np.ravel(hh_train.poor)
    # print(y_train)
    X_test = preprocess.pre_process_data(hh_test, enforce_cols=X_train.columns)

    pickle.dump(X_train, open(label + '_X_train.p', 'wb'))
    pickle.dump(y_train, open(label + '_y_train.p', 'wb'))
    pickle.dump(X_test, open(label + '_X_test.p', 'wb'))

    return X_train, y_train, X_test


def load_preprocessed_pickle(label):
    if os.path.isfile(label + '_X_train.p') and \
            os.path.isfile(label + '_y_train.p') and \
            os.path.isfile(label + '_X_test.p'):
        return pickle.load(open(label + '_X_train.p', 'rb')), \
               pickle.load(open(label + '_y_train.p', 'rb')), \
               pickle.load(open(label + '_X_test.p', 'rb'))
    else:
        return None, None, None


if __name__ == '__main__':
    main(sys.argv[1:])
