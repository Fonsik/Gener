import random
import numpy as np
import h5py


np.set_printoptions(linewidth=85)

h5f = h5py.File('Selection.h5', 'r')
Vector = h5f["Data"][:]
ID=h5f["ID"][:]

for i in range(20):
	print ID[i]
	print "--------------------------"
	print Vector[i]
	print "--------------------------"

