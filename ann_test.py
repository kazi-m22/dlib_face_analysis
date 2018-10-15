import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras import regularizers
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('facedata.csv', header = None)
data = data.values

x = data[:,0:6]
y = data[:,6]
# y = np.transpose(np.array([y]))
# x = np.expand_dims(x, axis=0)
# y = np.expand_dims(y, axis=0)
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.1)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model = Sequential()

model.add(Dense(10, input_dim=6, activation='relu'))
model.add(Dense(10, activation='relu'))
# model.add(Dense(5, activation='relu', kernel_regularizer=regularizers.l2(.1)))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

opt = Adam(lr = .1,decay=0.0)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit(X_train,y_train,epochs=1,validation_data=(X_test,y_test), batch_size=5)
model.save_weights('latest_best.h5')
test_ratim = [26.565,58.57,28.202,59.744,2.935661108,2.375]
test_me = [29.624,64.983,32.856,69.864,2.818548363,2.355078946]
test_me_trainset = X_train[1,:]
print(test_me_trainset)
print(model.predict(np.array([test_me_trainset,])))

