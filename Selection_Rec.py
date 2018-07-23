import tensorflow as tf
import keras
from keras.utils.io_utils import HDF5Matrix
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.callbacks import ModelCheckpoint
import h5py
import numpy as np

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

n=28
m=28
nump=50000
'''

x_train = HDF5Matrix('Selection.h5', 'Data', start=0, end=nump)
y_train = HDF5Matrix('Selection.h5', 'ID', start=0, end=nump)

x_test = HDF5Matrix('Selection.h5', 'Data', start=nump, end=2*nump)
y_test = HDF5Matrix('Selection.h5', 'ID', start=nump, end=2*nump)


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
'''


xr_train = HDF5Matrix('Tracks.h5', 'Data', start=0, end=nump/2)
yr_train = HDF5Matrix('Tracks.h5', 'Parameters', start=0, end=nump/2)

xr_test = HDF5Matrix('Tracks.h5', 'Data', start=nump/2, end=nump)
yr_test = HDF5Matrix('Tracks.h5', 'Parameters', start=nump/2, end=nump)

'''

xr_train=np.array(xr_train)
print xr_train.shape
#yr_train=np.array(yr_train)
#print yr_train.shape
xr_test=np.array(xr_test)
print xr_test.shape
#yr_test=np.array(yr_test)
#print yr_test.shape
'''

model2 = Sequential()
#model2.add(Conv2D(32,(3,3), input_shape=(n,m,1)))
#model2.add(P)
model2.add(Flatten(input_shape=(n,m)))
model2.add(Dense(784, activation='tanh', W_regularizer=l2(1e-5), input_dim=n*m))
model2.add(Dense(784, activation='tanh', W_regularizer=l2(1e-5), input_dim=n*m))
model2.add(Dense(784, activation='relu', W_regularizer=l2(1e-5), input_dim=n*m))
model2.add(Dense(784, activation='relu', W_regularizer=l2(1e-5), input_dim=n*m))
model2.add(Dense(784, activation='tanh', W_regularizer=l2(1e-5), input_dim=n*m))
model2.add(Dense(2, activation='linear'))


model2.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01))

keras.callbacks.ModelCheckpoint('chp2.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
'''
xr_train=xr_train.reshape(nump/2,n,m,1)
#yr_train=yr_train.reshape(nump/2,2,1)
xr_test=xr_test.reshape(nump/2,n,m,1)
#yr_test=yr_test.reshape(nump/2,2,1)
'''

model2.fit(xr_train, yr_train,
          epochs=100,
          batch_size=128,shuffle='batch')
score = model2.evaluate(xr_test, yr_test, batch_size=128)

model2.save('model_rec.h5')
model2.summary()

