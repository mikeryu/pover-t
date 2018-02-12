import numpy as np
import pandas as pd
from sklearn import preprocessing


# Convert all continuous-value features to the z-score of their respective populations
def standardize(df, numeric_only=True):
    numeric = df.select_dtypes(include=['int64', 'float64'])

    # subtract mean and divide by std
    df[numeric.columns] = (numeric - numeric.mean()) / numeric.std()

    return df

# Splits column in data frame with n possible values into n sub_columns.
# Value in original column becomes 1 in its new corresponding sub_columns,
# otherwise 0 in all other sub_columns.
def categorical_to_numerical(df):
    return pd.get_dummies(df)

# Ensure test set and training set columns match.
# drop all columns to dataframe that are not specified in enforce_cols
# and add all columns to dataframe that are specified in enforce_cols
# (any columns added to dataframe received empty/NaN values)
def enforce_columns(df, enforce_cols):
    to_drop = np.setdiff1d(df.columns, enforce_cols)
    to_add = np.setdiff1d(enforce_cols, df.columns)

    df.drop(to_drop, axis=1, inplace=True)
    df = df.assign(**{c: 0 for c in to_add})

    df.fillna(0, inplace=True)
    return df

def handle_missing_values(df):
    return df.interpolate()

# Standardize continuous-value data and convert categorical data to numerical
# Forces dataframe to have only columns present in enforce_cols
def pre_process_data(df, enforce_cols=None):
    print("Input shape:\t{}".format(df.shape))

    # print("NaN VALUES:")
    # print([num for num in df.isnull().sum() if num > 0])
    df = handle_missing_values(df)
    # print("NaN VALUES:")
    # print([num for num in df.isnull().sum() if num > 0])

    df = standardize(df)
    print("After standardization {}".format(df.shape))

    df = categorical_to_numerical(df)
    print("After converting categoricals:\t{}".format(df.shape))

    if enforce_cols is not None:
        df = enforce_columns(df, enforce_cols)

    return df

# The following is an attempt to move away from pandas early
# but probably isn't worth the trouble considering the the competition's
# starter code and sklearn's currently inconvenient categorical->numerical features
# See what would have made our lives easier -> http://scikit-learn.org/dev/modules/preprocessing.html#preprocessing-categorical-features

#Convert a pandas data frame to a set of examples,
#their corresponding labels, and the list of feature names
#Note: Drops the 'poor' and 'country' columns
# def df_to_examples(df):
#     ids = df.index.values
#     labels = df['poor'].values
#     mod_df = df.drop(columns=['poor', 'country'])
#     feature_sets = mod_df.values
#     feature_names = list(df)
#     return feature_names, feature_sets, labels, ids

# def _is_number(s):
#     try:
#         float(s)
#         return True
#     except ValueError:
#         return False
#
# def categorical_to_numerical(feature_sets):
#     category_mask = []
#     for feature in feature_sets[0]:
#         category_mask.append(not _is_number(feature))
#
#     label_encoder = preprocessing.LabelEncoder()
#     for feature_index in range(len(feature_sets[0])):
#         if category_mask[feature_index]:
#             label_encoder.fit_transform(zip(*feature_sets)[feature_index])
#     preprocessing.CategoricalEncoder()
#
#     one_hot_encoder = preprocessing.OneHotEncoder(categorical_features=category_mask)
#     return one_hot_encoder.fit_transform(feature_sets)

