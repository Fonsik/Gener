import random
import h5py
import numpy as np 
random.seed()

m=28
n=28
nump=20000

Vector=np.zeros((n*m),dtype=bool)
np.set_printoptions(linewidth=85)

h5f1 = h5py.File('tracks.h5', 'w')
dset=h5f1.create_dataset("Tracks", (nump*2,m*n))
pset=h5f1.create_dataset("Parameters", (nump, 2))
idset=h5f1.create_dataset("ID", (2*nump, 1), 'bool')


i=0
while (i<nump):
	a=random.uniform(-5,5)
	b=random.uniform(-30,30)
	par=np.array([a,b])
	i+=1
	ctr=0
	Vector=np.zeros((n*m),dtype=bool)
	for x in range (n):
		y=a*x+b
		y=int(y)
		y=m-1-y
		if (y<m and y>=0):
			Vector[m*y+x]=1
			ctr+=1
	if ctr<5:
		i-=1
	else:
		dset[i-1]=Vector
		idset[i-1]=1
		pset[i-1]=par
		if (i%1000 == 0):
			print "--- ... Processing event: ", i, "  ", round(100.0*((i+1)/float(3*nump)),2), "%" 


for i in range (nump):
	Vector=np.zeros((n*m),dtype=bool)
	r=random.uniform(5,25)
	r=int(r)
	for j in range (r):
		x=random.uniform(0,28)
		y=random.uniform(0,28)
		x=int(x)
		y=int(y)
		Vector[n*x+y]=1
	dset[i+nump]=Vector
	idset[i+nump]=0
	if (i%1000 == 0):
		print "--- ... Processing event: ", nump+i, "  ", round(100.0*((nump+i+1)/float(3*nump)),2), "%" 

h5f1.close()

h5f = h5py.File('data.h5', 'w')
dset=h5f.create_dataset("Tracks", (nump,m*n))
pset=h5f.create_dataset("Parameters", (nump, 2))


i=0
while (i<nump):
	a=random.uniform(-5,5)
	b=random.uniform(-30,30)
	par=np.array([a,b])
	i+=1
	ctr=0
	Vector=np.zeros((n*m),dtype=bool)
	r=random.uniform(5,15)
	r=int(r)
	for x in range (n):
		y=a*x+b
		y=int(y)
		y=m-1-y
		if (y<m and y>=0):
			Vector[m*y+x]=1
			ctr+=1
	if ctr>5:
		for j in range (r):
			x=random.uniform(0,28)
			y=random.uniform(0,28)
			x=int(x)
			y=int(y)
			Vector[n*x+y]=1
		dset[i-1]=Vector
		pset[i-1]=par
		if (i%1000 == 0):
			print "--- ... Processing event: ", 2*nump+i, "  ", round(100.0*((2*nump+i+1)/float(3*nump)),2), "%" 
	else:
		i-=1
	
h5f.close()

