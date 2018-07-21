import random
import h5py
import numpy as np 
random.seed()

m=28
n=28
nump=20000

Vector=np.zeros((n*m),dtype=bool)
np.set_printoptions(linewidth=85)
'''
h5f1 = h5py.File('tracks.h5', 'w')

i=0
while (i<nump):
	a=random.uniform(-5,5)
	b=random.uniform(-30,30)
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
		h5f1.create_dataset('dataset_'+str(i-1), data=Vector)
		h5f1.create_dataset('param_a'+str(i-1), data=a)
		h5f1.create_dataset('param_b'+str(i-1), data=b)
		if (i%1000 == 0):
			print "--- ... Processing event: ", i, "  ", round(100.0*((i+1)/float(3*nump)),2), "%" 

h5f1.close()

h5f2 = h5py.File('noise.h5', 'w')

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
	h5f2.create_dataset('noise_'+str(i), data=Vector)	
	if (i%1000 == 0):
		print "--- ... Processing event: ", nump+i, "  ", round(100.0*((nump+i+1)/float(3*nump)),2), "%" 

h5f2.close()

'''
h5f3 = h5py.File('data.h5', 'w')

i=0
while (i<nump):
	a=random.uniform(-3,3)
	b=random.uniform(0,28)
	i+=1
	ctr=0
	Vector=np.zeros((n*m),dtype=bool)
	Vector1=np.zeros((n*m),dtype=bool)
	Vector2=np.zeros((n*m),dtype=bool)
	r=random.uniform(5,25)
	r=int(r)
	for x in range (n):
		y=a*x+b
		y=int(y)
		y=m-1-y
		if (y<m and y>=0):
			Vector1[m*y+x]=1
			ctr+=1
	for j in range (r):
		x=random.uniform(0,28)
		y=random.uniform(0,28)
		x=int(x)
		y=int(y)
		Vector2[n*x+y]=1

	
	for x in range (n):
		for y in range (m):
			Vector[n*x+y]=Vector1[n*x+y]+Vector2[n*x+y]
	h5f3.create_dataset('dataset_'+str(i-1), data=Vector)
	h5f3.create_dataset('param_a'+str(i-1), data=a)
	h5f3.create_dataset('param_b'+str(i-1), data=b)
	if (i%1000 == 0):
		print "--- ... Processing event: ", 2*nump+i, "  ", round(100.0*((2*nump+i+1)/float(3*nump)),2), "%" 

h5f3.close()