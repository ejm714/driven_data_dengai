import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# def sorted_features(model, Xtrain_data):
#     sorted_features = sorted(zip(Xtrain_data.columns,model.feature_importances_), key = lambda x: x[1], reverse = True)
#     for feature in sorted_features:
#         print(feature)


def prep_x_and_y(data):
    """
    Takes in dataframe where first column is label and reamining are features.
    Must contain year, month, and day for sorting.
    Creates a test/train split chronologically; first 70 percent and last 30.
    """
    data = data.sort_values(['yr', 'month', 'day'])
    y = np.ravel(data.iloc[:, 0])
    X = data.iloc[:, 1:]
    # split chronologically
    split = round(data.shape[0]*.7)
    X_train = X.iloc[:split, :]
    X_test = X.iloc[split:, :]
    y_train = y[:split]
    y_test = y[split:]
    return X_train, X_test, y_train, y_test

def rf_feature_importances_single(data):
    """
    Takes in dataframe where first column is label and remaining are features.
    Creates a test/train split chronologically; first 70 percent and last 30.
    Prints mean absolute error and features importances.
    """
    X_train, X_test, y_train, y_test = prep_x_and_y(data)
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    y_pred=rf.predict(X_test)
    print("Mean absolute error: " + str(mean_absolute_error(y_test, y_pred)))
    print("")
    sorted_features = sorted(zip(X_train.columns,rf.feature_importances_), key = lambda x: x[1], reverse = True)
    for feature in sorted_features:
        print(feature)


def rf_feature_importances_by_city(data):
    """
    """
    sj = data[data.sj==1]
    iq = data[data.sj==0]

    y_pred = []
    y_true = []

    for df in [sj, iq]:
        X_train, X_test, y_train, y_test = prep_x_and_y(df)
        rf = RandomForestRegressor()
        rf.fit(X_train, y_train)
        pred=rf.predict(X_test)
        y_pred.extend(pred)
        y_true.extend(y_test)

        sorted_features = sorted(zip(X_train.columns,rf.feature_importances_), key = lambda x: x[1], reverse = True)
        for feature in sorted_features:
            print(feature)
        print("")

    print("Mean absolute error: " + str(mean_absolute_error(y_true, y_pred)))
