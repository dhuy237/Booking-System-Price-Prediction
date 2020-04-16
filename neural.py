import pandas as pd
from sklearn import preprocessing

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import regularizers

def print_result(model, x_val, y_val):
    y_pred_val = model.predict(x_val)
    print('---Test---')
    print('MAE: ', metrics.mean_absolute_error(y_val, y_pred_val))
    print('MSE: ', metrics.mean_squared_error(y_val, y_pred_val))
    print('R2: ', metrics.r2_score(y_val, y_pred_val))

df = pd.read_csv('./data/selected/data_selected_2.csv', index_col=0)

dataset = df.loc[:, df.columns != 'price']

X = dataset.loc[:, dataset.columns != 'city'].values

Y = df.loc[:, df.columns == 'price'].values

min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)

X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)

model = Sequential([
    Dense(100, activation='relu', input_shape=(159,)),
    Dense(100, activation='relu'),
    Dense(50, activation='relu'),
    Dense(10, activation='relu'),
    Dense(1, activation='linear'),
])

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

hist = model.fit(X_train, Y_train,
          batch_size=100, epochs=400,
          validation_data=(X_val, Y_val))

print(model.summary())

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

print_result(model, X_train, Y_train)
print_result(model, X_val, Y_val)