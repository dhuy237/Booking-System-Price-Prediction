import pandas as pd
import numpy as np

def feature_selected(df, coeffs):
    df = pd.DataFrame.drop(df, columns=['id', 'host_id'])
    
    numerics = ['int16','int32','int64','float16','float32','float64']
    numerical_vars = list(df.select_dtypes(include=numerics).columns)
    df = df[numerical_vars]

    col_set = []

    for i in range(len(coeffs)):
        if (coeffs[i]):
            col_set.append(df.columns[i])

    df = df[list(col_set)]

    return df

df = pd.read_csv('./data/data_cleaned_3.csv', index_col=0)

coeffs_4 = np.load('./data/selected_coefs_4.npy')

coeffs_4_1 = np.load('./data/selected_coefs_4_0_00001.npy')

coeffs_4_2 = np.load('./data/selected_coefs_4_0_0000001.npy')

coeffs_4_3 = np.load('./data/selected_coefs_4_0_00000001.npy')

df_selected = feature_selected(df, coeffs_4)
df_selected.to_csv('./data/selected/data_selected.csv')

df_selected_1 = feature_selected(df, coeffs_4_1)
df_selected_1.to_csv('./data/selected/data_selected_1.csv')

df_selected_2 = feature_selected(df, coeffs_4_2)
df_selected_2.to_csv('./data/selected/data_selected_2.csv')

df_selected_3 = feature_selected(df, coeffs_4_3)
df_selected_3.to_csv('./data/selected/data_selected_3.csv')

print(df_selected.shape)
print(df_selected_1.shape)
print(df_selected_2.shape)
print(df_selected_3.shape)