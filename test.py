import random
import numpy as np
import h5py

n=28
m=28
Vector=np.zeros((n*m),dtype=bool)
np.set_printoptions(linewidth=85)

h5f = h5py.File('data.h5', 'r')
Matrix = [[0 for x in range(n)] for y in range(m)] 

for i in range(20):
	Vector = h5f['dataset_'+str(i)][:]
	for x in range(n):
		for y in range(m):
			a=Vector[n*x+y]
			if a==1:
				Matrix[x][y]="*"
			else:
				Matrix[x][y]=0
			

	for x in range(n):
		print Matrix[x]
	print "--------------------------------------------------------"