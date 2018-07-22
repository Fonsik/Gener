import tensorflow as tf
import keras
from keras.utils.io_utils import HDF5Matrix
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense
import h5py
import numpy as np

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)
# Instantiating HDF5Matrix for the training set, which is a slice of the first 150 elements
n=28
m=28
nump=50000


x_train = HDF5Matrix('Selection.h5', 'Data', start=0, end=nump/2)
y_train = HDF5Matrix('Selection.h5', 'ID', start=0, end=nump/2)

x_test = HDF5Matrix('Selection.h5', 'Data', start=nump/2, end=nump)
y_test = HDF5Matrix('Selection.h5', 'ID', start=nump/2, end=nump)


# Define model
model = Sequential()
model.add(Dense(784, activation='relu', W_regularizer=l2(1e-5), input_dim=n*m))
model.add(Dense(784, activation='relu', W_regularizer=l2(1e-5), input_dim=n*m))
model.add(Dense(784, activation='relu', W_regularizer=l2(1e-5), input_dim=n*m))
model.add(Dense(784, activation='relu', W_regularizer=l2(1e-5), input_dim=n*m))
model.add(Dense(784, activation='relu', W_regularizer=l2(1e-5), input_dim=n*m))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=100,
          batch_size=128, shuffle='batch')
score = model.evaluate(x_test, y_test, batch_size=128)

model.save('model_selec.h5')
model.summary()