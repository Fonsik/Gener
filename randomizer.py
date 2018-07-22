import random
import numpy as np
import h5py

n=28
m=28
nump=20000

Vec=np.zeros((n*m),dtype=bool)

h5f1 = h5py.File('Training_data.h5', 'w')
dset=h5f1.create_dataset("Tracks", (nump*2,m*n), data=Vec)
idset=h5f1.create_dataset("ID", (2*nump, 1), 'bool')


h5f = h5py.File('tracks.h5', 'r')
Vector = h5f["Tracks"][:]
ID = h5f["ID"][:]

for i in range(2*nump):
	j=random.sample(range(2*nump), 2*nump)
	dset[i]=Vector[j]
	idset[i]=ID[j]

h5f1.close()