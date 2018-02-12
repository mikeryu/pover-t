
from sklearn import preprocessing

def preprocess():
    pass

#Convert a pandas data frame to a set of examples,
#their corresponding labels, and the list of feature names
#Note: Drops the 'poor' and 'country' columns
def df_to_examples(df):
    ids = df.index.values
    labels = df['poor'].values
    mod_df = df.drop(columns=['poor', 'country'])
    feature_sets = mod_df.values
    feature_names = list(df)
    return feature_names, feature_sets, labels, ids

def categorical_to_numerical():
    pass

def handle_missing_values():
    pass

