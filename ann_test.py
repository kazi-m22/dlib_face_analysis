import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


data = pd.read_csv('facedata.csv', header = None)
data = data.values

x = data[:,0:6]
y = data[:,6]
y = np.transpose(np.array([y]))
print(y.shape)

model = Sequential()

model.add(Dense(12, input_dim=6, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(4,activation='relu'))
model.add((Dense(1, activation='sigmoid')))

opt = Adam(lr = .0001,decay=0.0)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'],)

model.fit(x,y,epochs=50, batch_size=5)