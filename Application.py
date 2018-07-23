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

data = HDF5Matrix('data.h5', 'Tracks', start=0, end=nump)

h5f1 = h5py.File('output.h5', 'w')
dset=h5f1.create_dataset("Data", (nump,m,n))
psetpr=h5f1.create_dataset("Predicted parameters", (nump, 2))
psetrl=h5f1.create_dataset("Real parameters", (nump, 2))

h5f = h5py.File('data.h5', 'r')
Vector = h5f["Tracks"][:]
param=h5f["Parameters"][:]


model = load_model('model_rec.h5')
model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01))
'''
data=np.array(data)
data=data.reshape(50000,28,28,1)
'''
sc=model.predict(data)

for i in range(nump):
	dset[i]=data[i]
	psetpr[i]=sc[i]
	psetrl[i]=param[i]

h5f1.close()
