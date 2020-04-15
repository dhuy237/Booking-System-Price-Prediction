import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR


def print_result(model, x_val, y_val):
    y_pred_val = model.predict(x_val)
    print('---Test---')
    print('MAE: ', metrics.mean_absolute_error(y_val, y_pred_val))
    print('MSE: ', metrics.mean_squared_error(y_val, y_pred_val))
    print('R2: ', metrics.r2_score(y_val, y_pred_val))

def Linear(x_train, y_train):
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model

def svr(x_train, y_train):
    regressor = SVR(kernel='rbf', max_iter=5000, verbose=1)
    regressor.fit(x_train, y_train)
    return regressor

if __name__ == "__main__":
    # Linear Regression model

    # df = pd.read_csv('./data/selected/data_selected.csv')
    # dataset = df.loc[:, df.columns != 'price']

    # X = dataset.loc[:, dataset.columns != 'city'].values
    # Y = df.loc[:, df.columns == 'price'].values

    # min_max_scaler = preprocessing.MinMaxScaler()
    # X_scale = min_max_scaler.fit_transform(X)

    # X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)
    # X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

    # linear_model = Linear(X_train, Y_train)
    # print_result(linear_model, X_train, Y_train)
    # print_result(linear_model, X_val, Y_val)
    # print_result(linear_model, X_test, Y_test)


    # SVR
    
    df = pd.read_csv('./data/selected/data_selected_3.csv')
    dataset = df.loc[:, df.columns != 'price']

    X = dataset.loc[:, dataset.columns != 'city'].values
    Y = df.loc[:, df.columns == 'price'].values

    sc_X = preprocessing.StandardScaler()
    sc_Y = preprocessing.StandardScaler()
    X = sc_X.fit_transform(X)
    Y = sc_Y.fit_transform(Y)

    X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X, Y, test_size=0.3)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

    svr_model = svr(X_train, Y_train.ravel())
    print_result(svr_model, X_train, Y_train)
    print_result(svr_model, X_val, Y_val)
    print_result(svr_model, X_test, Y_test)

