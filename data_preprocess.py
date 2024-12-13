import pandas as pd
import numpy as np

def read_data():
    train = pd.read_csv("../../kaggle_data/train.csv")
    test = pd.read_csv("../../kaggle_data/test.csv")
    return train, test

def process_na(train, test):
    train = train.loc[np.all(train.notna(), axis=1), :]
    test.loc[test['parentspecies'].isna(), 'parentspecies'] = 'toluene'
    return train, test

def one_hot_encode(train, test):
    train = train.join(pd.get_dummies(train['parentspecies'], dtype=int))
    train = train.drop('parentspecies', axis=1)
    test = test.join(pd.get_dummies(test['parentspecies'], dtype=int))
    test = test.drop('parentspecies', axis=1)
    test.loc[:, 'decane_toluene'] = 0
    return train, test

def split_x_y(train, test, dropped_features):
    train_y = train['log_pSat_Pa']
    train_x = train.loc[:, ~train.columns.isin(dropped_features)]
    test_x = test.loc[:, ~test.columns.isin(dropped_features)]
    return train_x, train_y, test_x

def load_and_preprocess_data(dropped_features = ['ID']):
    (train, test) = read_data()
    (train, test) = process_na(train, test)
    (train, test) = one_hot_encode(train, test)
    return split_x_y(train, test, dropped_features)