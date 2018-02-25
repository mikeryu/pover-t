import pandas as pd


# Creates a dataframe for the specified country and predicted probabilities
def create_prediction_dataframe(preds, test_feat, country):
    country_sub = pd.DataFrame(data=preds[:, 1],  # proba p=1
                               columns=['poor'],
                               index=test_feat.index)

    # add the country code as a column to distinguish between countries
    # once the dataframes are collapsed into a single dataframe
    country_sub["country"] = country
    return country_sub[["country", "poor"]]


# Finds the prediction probabilities of each country's test set on each model
# and stores that information in a dataframe for output in competition format
def convert_prediction_to_dataframe(model_a, model_b, model_c, aX_test, bX_test, cX_test):
    models_tests = ((model_a, aX_test, 'A'), (model_b, bX_test, 'B'), (model_c, cX_test, 'C'))
    return pd.concat(
        [convert_prediction_to_dataframe_singular(model, X_test, lbl) for model, X_test, lbl in models_tests])


def convert_prediction_to_dataframe_singular(model, X_test, label):
    predictions = model.predict_proba(X_test)
    df = create_prediction_dataframe(predictions, X_test, label)
    return df
