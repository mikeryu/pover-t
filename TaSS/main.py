

from TaSS import load_data
from TaSS import preprocess
from TaSS import learn

hh_train, hh_test = load_data.read_household_data()
indiv_train, indiv_test = load_data.read_individual_data()
# print(hh_train[0].head())
# print(hh_test[1].head())
# print(indiv_train[0].head())
# print(indiv_train[2].head())

# print(hh_train[0].columns)
# print(hh_train[0]['country'])

print(preprocess.df_to_examples(hh_train[0].head()))

