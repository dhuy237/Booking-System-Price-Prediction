import pandas as pd
import numpy as np
import xgboost
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def print_result(model, x_val, y_val):
    y_pred_val = model.predict(x_val)
    print('---Test---')
    print('MAE: ', metrics.mean_absolute_error(y_val, y_pred_val))
    print('MSE: ', metrics.mean_squared_error(y_val, y_pred_val))
    print('R2: ', metrics.r2_score(y_val, y_pred_val))

def XGB(x_train, y_train):
    xgb = xgboost.XGBRegressor(learning_rate=0.01, n_estimators=1000, max_depth=12,
                                subsample=0.8, colsample_bytree=0.3, gamma=1)

    xgb.fit(x_train, y_train, verbose=True)
    return xgb

if __name__ == "__main__":
    df = pd.read_csv('./data/selected/data_selected_3.csv')

    dataset = df.loc[:, df.columns != 'price']

    X = dataset.loc[:, dataset.columns != 'city'].values
    Y = df.loc[:, df.columns == 'price'].values

    X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X, Y, test_size=0.3)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

    xgb = XGB(X_train, Y_train)
    print_result(xgb, X_train, Y_train)
    print_result(xgb, X_val, Y_val)
    print_result(xgb, X_test, Y_test)