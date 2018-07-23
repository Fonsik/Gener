import random
import numpy as np
import h5py
import matplotlib.pyplot as plt
from PIL import Image

n=28
m=28
nump=50000
np.set_printoptions(linewidth=90)

h5f = h5py.File('output.h5', 'r')
paramp = h5f["Predicted parameters"][:]
paramr=h5f["Real parameters"][:]


h5f = h5py.File('data.h5', 'r')
Vector = h5f["Tracks"][:]
param=h5f["Parameters"][:]
'''
for i in range(20):
	plt.imshow(Vector[i], interpolation='nearest')
	plt.show()
'''

print paramp[0]
print paramr[0]
ap=np.zeros((50000))
ar=np.zeros((50000))
bp=np.zeros((50000))
br=np.zeros((50000))
for i in range(50000):
	ap[i]=paramp[i][0]
	ar[i]=paramr[i][0]
	bp[i]=paramp[i][1]
	br[i]=paramr[i][1]




plt.plot(ar, ap, 'ro')
plt.show()

plt.plot(br, bp, 'ro')
plt.show()
	