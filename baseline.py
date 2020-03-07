import numpy as np 
import pandas as pd 
import sklearn as sklearn

from sklearn import linear_model 
from sklearn import feature_selection 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans

from sklearn.linear_model import Lasso 
from sklearn.linear_model import Ridge

import multiprocessing

'''Function to print metrics for validation set'''
def print_result_val(trained_model, trained_model_name, x_test, y_test):
    print('--------- For Model: ', trained_model_name, ' ---------\n')
    predicted_values = trained_model.predict(x_test)
    print("Mean absolute error: ",
          metrics.mean_absolute_error(y_test, predicted_values))
    # print("Median absolute error: ",
    #       metrics.median_absolute_error(y_test, predicted_values))
    print("Mean squared error: ", metrics.mean_squared_error(
        y_test, predicted_values))
    print("R2: ", metrics.r2_score(y_test, predicted_values))


'''Function to print metrics for train set'''
def print_result_train(trained_model, trained_model_name, x_test, y_test):
    print('--------- For Model: ', trained_model_name, ' --------- (Train Data)\n')
    predicted_values = trained_model.predict(x_test)
    print("Mean absolute error: ",
          metrics.mean_absolute_error(y_test, predicted_values))
    # print("Median absolute error: ",
    #       metrics.median_absolute_error(y_test, predicted_values))
    print("Mean squared error: ", metrics.mean_squared_error(
        y_test, predicted_values))
    print("R2: ", metrics.r2_score(y_test, predicted_values))

def Linear(x_train, y_train, x_val, y_val): 
    print("Begin Linear")
    lireg = linear_model.LinearRegression(n_jobs = 4).fit(x_train, y_train)
    print_result_val(lireg, "Linear Regression", x_val, y_val.ravel())
    print("End Linear")
    return

def Linear_Lasso(x_train, y_train, x_val, y_val): 
    print("Begin Linear with Lasso")
    lasso = Lasso(alpha = 0.3)
    lasso.fit(x_train, y_train)
    print_result_val(lasso, "Linear Regression with Lasso Regularization", x_val, y_val)
    print("End Linear")
    return

def Linear_Ridge(x_train, y_train, x_val, y_val): 
    print("Begin Linear with Ridge")

    ridge = Ridge(alpha = 8)
    ridge.fit((x_train, y_train))
    y_pred = rdige.predict(x_val.values)

    print('Coefficients: \n', ridge.coef_)
    print("MSE: %.2f" % mean_squared_error(y_val, y_pred))
    print('Variance_score: %.2f' % r2_score(y_val, y_pred))
    print("R2 score: ", sklearn.metrics.r2_score(y_val, y_pred))
    
    print(ridge, "Linear Regression with Ridge Regression", x_val, y_val)
    print("End Linear")
    return

if __name__ == "__main__": 
    x_train = pd.read_csv('london/data_clean/data_cleaned_train_comments_X.csv')
    y_train = pd.read_csv('london/data_clean/data_cleaned_train_y.csv')

    x_val = pd.read_csv('london/data_clean/data_cleaned_val_comments_X.csv')
    y_val = pd.read_csv('london/data_clean/data_cleaned_val_y.csv')

    x_test = pd.read_csv('london/data_clean/data_cleaned_test_comments_X.csv')
    y_test = pd.read_csv('london/data_clean/data_cleaned_test_y.csv')

    coeffs = np.load('london/data_clean/cv_coeffs.npy')
    column_set = set()

    feat_ls = [
        'host_identity_verified',
    'latitude',
    'longitude',
    'accommodates',
    'bathrooms',
    'bedrooms',
    'beds',
    'guests_included',
    'security_deposit',
    'cleaning_fee',
    'extra_people',
    'number_of_reviews',
    'review_scores_rating',
    'review_scores_accuracy',
    'review_scores_cleanliness',
    'review_scores_location',
    'review_scores_value',
    'reviews_per_month',
    'comments'

    ]

    for i in range(len(coeffs)):
        if (coeffs[i]): 
            column_set.add(x_train.columns[i])
    x_train = x_train[list(column_set)]
    x_val = x_val[list(column_set)]
    x_test = x_test[list(column_set)]

    x_concat = pd.concat([x_train, x_val], ignore_index = True) 
    y_concat = pd.concat([y_train, y_val], ignore_index = True)

    print("--------------------Linear Regression--------------------")
    Linear(x_concat, y_concat, x_test, y_test)

    print("--------------------Linear Regression with Lasso--------------------")
    Linear_Lasso(x_train, y_train, x_val, y_val)

    print("--------------------Linear Regression with Ridge--------------------")
    Linear_Ridge(x_concat, y_concat, x_test, y_test)