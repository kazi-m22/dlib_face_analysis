import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


ratim_data = pd.read_csv('features_ratim.csv', header=None)
ratim_data = ratim_data.values
mydata = pd.read_csv('features.csv', header=None)
mydata = mydata.values

ratim_x = ratim_data[:,0:6]
my_x = mydata[:372,0:6]

ratim_data_shape = tuple(ratim_x.shape)
my_data_shape = tuple(my_x.shape)

ratim_lip_ratio = ratim_x[:,4]
my_lip_ratio = my_x[:,4]
#
# fig = plt.figure()
# plot = fig.add_subplot(111)
# plot.scatter(np.array([list(range(0,ratim_data_shape[0]))]), np.array([ratim_lip_ratio[:]]))
# plot.scatter(np.array([list(range(0,my_data_shape[0]))]), np.array(my_lip_ratio[:]))
# fig.show()

plt.subplot(221)
plt.plot(list(range(0,ratim_data_shape[0])), list(ratim_lip_ratio[:]), 'b')
plt.plot(list(range(0,my_data_shape[0])), list(my_lip_ratio[:]), 'r')
plt.ylim((0,20))
plt.title('lip ratio comparison')
plt.show()

plt.subplot(222)
ratim_eye_ratio = ratim_x[:,5]
my_eye_ratio = my_x[:,5]
plt.plot(list(range(0,ratim_data_shape[0])), list(ratim_eye_ratio[:]), 'b')
plt.plot(list(range(0,my_data_shape[0])), list(my_eye_ratio[:]), 'r')
plt.title('eye ratio')
plt.ylim(0,5)
plt.show()
