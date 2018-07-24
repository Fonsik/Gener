import random
import h5py
import numpy as np 
random.seed()

m=28
n=28
nump=50000
'''
Vector=np.zeros((n*m),dtype=int)
np.set_printoptions(linewidth=85)

h5f1 = h5py.File('Selection.h5', 'w')
h5f2 = h5py.File('Tracks.h5', 'w')

dset=h5f1.create_dataset("Data", (nump*2,m*n))
idset=h5f1.create_dataset("ID", (2*nump, 1), 'i')


dset2=h5f2.create_dataset("Data", (nump,m*n))
pset=h5f2.create_dataset("Parameters", (nump, 2))



i=0
rali=random.sample(range(2*nump), 2*nump)
while (i<nump):
	a=random.uniform(-5,5)
	b=random.uniform(-30,30)
	par=np.array([a,b])
	i+=1
	ctr=0
	Vector=np.zeros((n*m),dtype=int)
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
		j=rali[i-1]
		dset[j]=Vector
		idset[j]=1
		dset2[i-1]=Vector
		pset[i-1]=par
		j=rali[i+nump-1]
		Vector=np.zeros((n*m),dtype=int)
		r=random.uniform(5,25)
		r=int(r)
		for k in range (r):
			x=random.uniform(0,28)
			y=random.uniform(0,28)
			x=int(x)
			y=int(y)
			Vector[n*x+y]=1
		dset[j]=Vector
		idset[j]=0
		if (i%500 == 0):
			print "--- ... Processing event: ", 2*i, "  ", round(100.0*((2*i+1)/float(3*nump)),2), "%" 

h5f1.close()
h5f2.close()
'''
h5f = h5py.File('Tracks.h5', 'w')
dset=h5f.create_dataset("Data", (nump,n,m))
pset=h5f.create_dataset("Parameters", (nump, 2))


i=0
while (i<nump):
	a=random.uniform(-10,10)
	b=random.uniform(-30,30)
	par=np.array([a,b])
	i+=1
	ctr=0
	Vector=np.zeros((n,m),dtype=int)
	r=random.uniform(5,15)
	eta=random.uniform(0,1)
	r=int(r)
	for x in range (n):
		y=a*x+b
		y=int(y)
		y=m-1-y
		if (y<m and y>=0 and eta<0.8):
			Vector[y][x]=1
			ctr+=1
	if ctr>6:
		for j in range (n):
			for k in range(m):
				p=random.uniform(0,1)
				if p<0.05:
					Vector[j][k]=1
		dset[i-1]=Vector
		pset[i-1]=par
		if (i%1000 == 0):
			print "--- ... Processing event: ", i, "  ", round(100.0*((i+1)/float(2*nump)),2), "%" 
	else:
		i-=1
	
h5f.close()


h5f = h5py.File('data.h5', 'w')
dset=h5f.create_dataset("Tracks", (nump,n,m))
pset=h5f.create_dataset("Parameters", (nump, 2))


i=0
while (i<nump):
	a=random.uniform(-10,10)
	b=random.uniform(-30,30)
	par=np.array([a,b])
	i+=1
	ctr=0
	Vector=np.zeros((n,m),dtype=int)
	r=random.uniform(5,15)
	eta=random.uniform(0,1)
	r=int(r)
	for x in range (n):
		y=a*x+b
		y=int(y)
		y=m-1-y
		if (y<m and y>=0 and eta<0.8):
			Vector[y][x]=1
			ctr+=1
	if ctr>6:
		for j in range (n):
			for k in range(m):
				p=random.uniform(0,1)
				if p<0.05:
					Vector[j][k]=1
		dset[i-1]=Vector
		pset[i-1]=par
		if (i%1000 == 0):
			print "--- ... Processing event: ", nump+i, "  ", round(100.0*((nump+i+1)/float(2*nump)),2), "%" 
	else:
		i-=1
	
h5f.close()