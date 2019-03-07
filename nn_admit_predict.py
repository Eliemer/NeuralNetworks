import numpy as np
import pandas as pd
import time
import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
from keras.models import Sequential
from keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

# import data

data = pd.read_csv("Datasets/Admission_Predict_Ver1.1.csv")

y = data['Chance of Admit '].values.reshape(-1, 1)
X = data.drop(['Chance of Admit ', 'Serial No.'], axis=1)

scalerX = MinMaxScaler()
scalerY = MinMaxScaler()

X_scaled = scalerX.fit_transform(X)
y_scaled = scalerY.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled)

# build model

n_cols = X_scaled.shape[1]
model = Sequential()
model.add(Dense(12, activation='relu', input_shape=(n_cols, )))
model.add(Dense(12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mse', 'mae'])

history = model.fit(X_train,
                    y_train,
                    epochs=15,
                    validation_split=0.2,
                    shuffle=True,
                    verbose=0)

# Graph history

# import matplotlib.pyplot as plt
#
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()

# predict

Xnew = np.array([[335, 105, 4, 4, 5, 8.2, 1]])

ynew = scalerY.inverse_transform(model.predict(scalerX.transform(Xnew)))
print("\nX=%s\n Predicted=%s\n\n" % (Xnew[0], ynew[0]))
