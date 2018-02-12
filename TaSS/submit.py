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
def convert_prediction_to_dataframe(model_a, aX_test, model_b, bX_test, model_c, cX_test):
    a_preds = model_a.predict_proba(aX_test)
    b_preds = model_b.predict_proba(bX_test)
    c_preds = model_c.predict_proba(cX_test)

    a_df = create_prediction_dataframe(a_preds, aX_test, 'A')
    b_df = create_prediction_dataframe(b_preds, bX_test, 'B')
    c_df = create_prediction_dataframe(c_preds, cX_test, 'C')

    return pd.concat([a_df, b_df, c_df])
