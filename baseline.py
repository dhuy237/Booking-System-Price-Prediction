import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import metrics

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


if __name__ == "__main__":
    x_train = pd.read_csv('data_cleaned_train_X.csv')
    y_train = pd.read_csv('data_cleaned_train_y.csv')

    x_val = pd.read_csv('data_cleaned_val_X.csv')
    y_val = pd.read_csv('data_cleaned_val_y.csv')

    x_test = pd.read_csv('data_cleaned_test_X.csv')
    y_test = pd.read_csv('data_cleaned_test_y.csv')

    # Linear Regression model
    linear_model = Linear(x_train, y_train)
    print_result(linear_model, x_train, y_train)
    print_result(linear_model, x_val, y_val)
    print_result(linear_model, x_test, y_test)

    