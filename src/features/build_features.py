import pandas as pd
import numpy as np
import pickle

def preprocess(data_path, labels_path=None):
    features = pd.read_csv(data_path)
    if labels_path:
        labels = pd.read_csv(labels_path)
        data = pd.merge(labels, features, how='left', on=['city', 'year', 'weekofyear'])
        return data
    else:
        results_cols = features[['city', 'year', 'weekofyear']]
        return results_cols, features,


def create_features(data, filename):
    # city dummy
    data['sj'] = data['city'].map(lambda x: 1 if x=='sj' else 0)
    data.drop('city', axis=1, inplace = True)

    # date parsing
    data['date'] = pd.to_datetime(data.week_start_date)
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data['yr'] = data['date'].dt.year
    data.drop(['year', 'weekofyear', 'week_start_date', 'date'], axis=1, inplace = True)

    data = data.interpolate()

    data.to_pickle('../data/processed/' + filename + '.pkl')

    return data


# df = preprocess('../../data/raw/dengue_features_train.csv', '../../data/raw/dengue_labels_train.csv')
# df = create_features(df, 'train')
#
# results_cols, test_df = preprocess('../../data/raw/dengue_features_test.csv')
# test_df = create_features(df, 'test')
