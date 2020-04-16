import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor

def print_result(model, x_val, y_val):
    y_pred_val = model.predict(x_val)
    print('---Test---')
    print('MAE: ', metrics.mean_absolute_error(y_val, y_pred_val))
    print('MSE: ', metrics.mean_squared_error(y_val, y_pred_val))
    print('R2: ', metrics.r2_score(y_val, y_pred_val))

def RandomForest(x_train, y_train):
    forest_model = RandomForestRegressor(n_estimators=50, n_jobs=-1, max_features='sqrt', max_depth=50, verbose=1)
    forest_model.fit(x_train, y_train)
    return forest_model

def Bagging(x_train, y_train):
    bag = BaggingRegressor(n_estimators=50, n_jobs=-1, verbose=1)
    bag.fit(x_train, y_train)
    return bag

def GradientBoosting(x_train, y_train):
    grad = GradientBoostingRegressor(n_estimators=1000, verbose=1)
    grad.fit(x_train, y_train)
    return grad

if __name__ == "__main__":
    df = pd.read_csv('./data/selected/data_selected_3.csv')

    dataset = df.loc[:, df.columns != 'price']

    X = dataset.loc[:, dataset.columns != 'city'].values
    Y = df.loc[:, df.columns == 'price'].values

    X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X, Y, test_size=0.3)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

    forest = RandomForest(X_train, Y_train.ravel())
    print_result(forest, X_train, Y_train)
    print_result(forest, X_val, Y_val)
    print_result(forest, X_test, Y_test)

    # bag = Bagging(X_train, Y_train.ravel())
    # print_result(bag, X_train, Y_train)
    # print_result(bag, X_val, Y_val)
    # print_result(bag, X_test, Y_test)

    # grad = GradientBoosting(X_train, Y_train.ravel())
    # print_result(grad, X_train, Y_train)
    # print_result(grad, X_val, Y_val)
    # print_result(grad, X_test, Y_test)