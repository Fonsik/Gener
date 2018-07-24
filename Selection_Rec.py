import tensorflow as tf
import keras
import random
from keras.utils.io_utils import HDF5Matrix
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, Reshape
from keras.callbacks import ModelCheckpoint
import h5py
import numpy as np
import matplotlib.pyplot as plt
from keras_sequential_ascii import sequential_model_to_ascii_printout

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

n=28
m=28
nump=50000

xr_train=np.load('x_train.npy')
yr_train=np.load('y_train.npy')
xr_test=np.load('x_test.npy')
yr_test=np.load('y_test.npy')



model2 = Sequential()
input_shape=(1,n,m)

model2.add(Reshape((n,m,1),input_shape=input_shape))
model2.add((Flatten()))
model2.add(Dense(784, activation='tanh', W_regularizer=l2(1e-5), input_dim=n*m))
model2.add(Dense(784, activation='tanh', W_regularizer=l2(1e-5), input_dim=n*m))
model2.add(Dense(784, activation='relu', W_regularizer=l2(1e-5), input_dim=n*m))
model2.add(Dense(784, activation='relu', W_regularizer=l2(1e-5), input_dim=n*m))
model2.add(Dense(784, activation='tanh', W_regularizer=l2(1e-5), input_dim=n*m))
model2.add(Dense(2, activation='linear'))
model2.add(Reshape((1, 2),input_shape=(2,)))


model2.compile(loss='mean_squared_error', optimizer='adam')

keras.callbacks.ModelCheckpoint('chp2.h5', monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

sequential_model_to_ascii_printout(model2)

history=model2.fit(xr_train, yr_train,
          epochs=12,
          batch_size=128,shuffle='batch')
score = model2.evaluate(xr_test, yr_test, batch_size=128)

print('Test loss:', score)

model2.save('model_test.h5')
print (model2.summary())
'''
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
'''