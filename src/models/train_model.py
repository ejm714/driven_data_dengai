import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, make_scorer

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
    return X, y, X_train, X_test, y_train, y_test

def rf_feature_importances_single(data):
    """
    Takes in dataframe where first column is label and remaining are features.
    Creates a test/train split chronologically; first 70 percent and last 30.
    Prints mean absolute error and features importances.
    """
    X, y, X_train, X_test, y_train, y_test = prep_x_and_y(data)
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
        X, y, X_train, X_test, y_train, y_test = prep_x_and_y(df)
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


def run_tree_models(xtrain_data, xtest_data, ytrain_data, ytest_data):
    trees = {}

    trees['dt'] = DecisionTreeRegressor(random_state=42)
    trees['extratrees'] = ExtraTreeRegressor(random_state=42)
    trees['randomForest'] = RandomForestRegressor(random_state=42)
    trees['bagged_randomForest'] = BaggingRegressor(RandomForestRegressor(random_state=42))
    trees['adaboostedTrees'] = AdaBoostRegressor(random_state=42)
    trees['gradboostedTrees'] = GradientBoostingRegressor(random_state=42)

    for name,model in trees.items():
        model.fit(xtrain_data, ytrain_data)
        y_pred=model.predict(xtest_data)
        print('Model: ' + name)
        print("Mean absolute error: " + str(mean_absolute_error(ytest_data, y_pred)))
        print("")


def run_grid_search_rf(X, y, xtrain_data, xtest_data, ytrain_data, ytest_data):
    rf = RandomForestRegressor(random_state=42)
    # can set these as a function param if desired in future
    params = {'n_estimators': [10, 20, 50, 100], 'min_samples_split': [2, 5, 10, 20], 'min_samples_leaf': [1, 2, 10], 'max_depth': [1, 2, 5, 10, 50]}
    # cross validated grid search so using entire data rather than train/test
    X = pd.concat([xtrain_data, xtest_data], axis = 0)
    y = np.concatenate((ytrain_data, ytest_data))

    grid_dict = GridSearchCV(rf, params, verbose = 1, scoring = make_scorer(mean_absolute_error), n_jobs = 2)
    train_fit = grid_dict.fit(X, y)
    print('Best Score: %0.3f' % train_fit.best_score_)
    print('Optimal Parameters: ', train_fit.best_params_)
    return train_fit.best_params_


def calculate_mae_for_train(best_params, xtrain_data, xtest_data, ytrain_data, yest_data):
    final_model = RandomForestRegressor(**best_params, random_state=42)
    final_model.fit(xtrain_data, ytrain_data)
    y_pred = final_model.predict(xtest_data)
    print("Mean absolute error: " + str(mean_absolute_error(yest_data, y_pred)))


def fit_final_model(best_params, X, y):
    final_model = RandomForestRegressor(**best_params, random_state=42)
    # refit on entire data
    fitted_model = final_model.fit(X, y)
    y_pred = fitted_model.predict(X)
    print("Mean absolute error: " + str(mean_absolute_error(y, y_pred)))
    return fitted_model


def create_results_df(model, data, X):
    results_df = data[['total_cases', 'sj']]
    # predict
    results_df['predictions'] = model.predict(X)
    return results_df



# X, y, X_train, X_test, y_train, y_test = prep_x_and_y(df[df.sj==0])
# iq_best_params = run_grid_search_rf(X, y, X_train, X_test, y_train, y_test)
# iq_model = fit_final_model(iq_best_params, X, y)
#
# X, y, X_train, X_test, y_train, y_test = prep_x_and_y(df[df.sj==1])
# sj_best_params = run_grid_search_rf(X, y, X_train, X_test, y_train, y_test)
# sj_model = fit_final_model(sj_best_params, X, y)
#
# joblib.dump(iq_model, '../../models/iq_model.pkl')
# joblib.dump(sj_model, '../../models/sj_model.pkl')
