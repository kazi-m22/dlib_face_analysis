import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split


data = pd.read_csv('features.csv', header = None)
data = data.values

x = data[:,0:6]
y = data[:,6]
y = np.transpose(np.array([y]))

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.1)

model = Sequential()

model.add(Dense(6, input_dim=6, activation='relu'))

model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))


model.add(Dense(1, activation='sigmoid'))

opt = Adam(lr = .00001,decay=0.0)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit(X_train,y_train,epochs=20,validation_data=(X_test,y_test), batch_size=5)