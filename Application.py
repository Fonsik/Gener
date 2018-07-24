import random
import numpy as np
import h5py
from keras.utils.io_utils import HDF5Matrix
from keras.models import load_model
from keras.regularizers import l2
from keras.optimizers import SGD

nump=50000
n=28
m=28

data=np.load('data.npy')

model = load_model('cnn.h5')
model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01))

model2 = load_model('model_test.h5')
model2.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01))


sc1=model.predict(data)
sc2=model.predict(data)

np.save("cnn_out", sc1)
np.save("dense_out", sc2)
