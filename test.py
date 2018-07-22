import random
import numpy as np
import h5py


np.set_printoptions(linewidth=85)

h5f = h5py.File('tracks.h5', 'r')
h5f2 = h5py.File('data.h5', 'r')

Vector = h5f["Tracks"][:]
param = h5f2["Tracks"][:]


for i in range(20):
	print param[i]
	print "--------------------------"
	print Vector[i]
	print "--------------------------"
	print Vector[i+20000]
	print "--------------------------"

