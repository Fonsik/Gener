from keras.models import Sequential
from keras.layers import Dense
import h5py
import numpy as np


# Instantiating HDF5Matrix for the training set, which is a slice of the first 150 elements
n=28
m=28
nump=20000

h5f = h5py.File('tracks.h5', 'r')
Vector = h5f["Tracks"][:]
ID = h5f["ID"][:]


'''
X_train = HDF5Matrix('tracks.h5', 'Tracks', start=0, end=150)
y_train = HDF5Matrix('tracks.h5', 'ID', start=0, end=150)

X_test = HDF5Matrix('tracks.h5', 'Tracks', start=150, end=200)
y_test = HDF5Matrix('tracks.h5', 'ID', start=150, end=200)
'''



# Define model
model = Sequential()
model.add(Dense(784, activation='relu', W_regularizer=l2(1e-5), input_dim=n*m))
model.add(Dense(784, activation='relu', W_regularizer=l2(1e-5), input_dim=n*m))
model.add(Dense(784, activation='relu', W_regularizer=l2(1e-5), input_dim=n*m))
model.add(Dense(2, activation='softmax'))

# Set loss and optimizer
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.01), metrics=['accuracy', ])



# Note: you have to use shuffle='batch' or False with HDF5Matrix
model.fit(Training, IDT, batch_size=32, shuffle='batch')

model.evaluate(Test, idt, batch_size=32)

model.save('model_test.h5')
model.summary()