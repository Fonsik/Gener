import random
import h5py
import numpy as np
import pickle 
random.seed()

m=28
n=28
nump=2000

Vector=np.zeros((n*m),dtype=bool)
np.set_printoptions(linewidth=85)
Matrix=np.zeros((nump))

i=0
while (i<nump):
	a=random.uniform(-3,3)
	b=random.uniform(0,28)
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
		with h5py.File('tracks.h5', 'w') as f:
			f.create_dataset("tracks",  data=Vector)

f.close()


'''
noise = ROOT.TFile("noise.root","RECREATE")
ns = TTree( 'nis', 'nis' )

vecttab=[]
ss=[]

for x in range(n):
	for y in range (m):
		vecttab.append(array.array('i', [0]))
		ss.append("a"+str(x)+str(y))
		ns.Branch(ss[n*x+y],vecttab[-1], ss[n*x+y]+"/I")

for i in range (nump):
	Matrix = [[0 for x in range(n)] for y in range(m)] 
	r=random.uniform(0,50)
	r=int(r)
	for j in range (r):
		x=random.uniform(0,28)
		y=random.uniform(0,28)
		x=int(x)
		y=int(y)
		Matrix[x][y]=1
	for x in range (n):
		for y in range (m):
			vecttab[n*x+y][0]=Matrix[x][y]	
	ns.Fill()
	if (i%1000 == 0):
		print "--- ... Processing event: ", nump+i, "  ", round(100.0*((nump+i+1)/float(3*nump)),2), "%" 


noise.Write()
noise.Close()


Matrix1 = [[0 for x in range(n)] for y in range(m)] 
Matrix2 = [[0 for x in range(n)] for y in range(m)] 

data = ROOT.TFile("data.root","RECREATE")
d = TTree( 'dat', 'dat' )

vecttab=[]
ss=[]

for x in range(n):
	for y in range (m):
		vecttab.append(array.array('i', [0]))
		ss.append("a"+str(x)+str(y))
		d.Branch(ss[n*x+y],vecttab[-1], ss[n*x+y]+"/I")


i=0
while (i<nump):
	a=random.uniform(-3,3)
	b=random.uniform(0,28)
	i+=1
	ctr=0
	Matrix1 = [[0 for x in range(n)] for y in range(m)] 
	Matrix2 = [[0 for x in range(n)] for y in range(m)]
	r=random.uniform(0,50)
	r=int(r)
	for x in range (n):
		y=(n-b-x)/a
		y=int(y)
		if (y<m and y>=0):
			Matrix1[x][y]=1
			ctr+=1
	for j in range (r):
		x=random.uniform(0,28)
		y=random.uniform(0,28)
		x=int(x)
		y=int(y)
		Matrix2[x][y]=1

	
	for x in range (n):
		for y in range (m):
			temp=Matrix1[x][y]+Matrix2[x][y]
			if (temp==1 or temp==2):
				vecttab[n*x+y][0]=1
			else:
				vecttab[n*x+y][0]=0
	d.Fill()

	if (i%1000 == 0):
		print "--- ... Processing event: ", 2*nump+i, "  ", round(100.0*((2*nump+i+1)/float(3*nump)),2), "%" 

data.Write()
data.Close()
'''