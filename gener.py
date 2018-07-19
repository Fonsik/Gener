import random
from termcolor import colored
import ROOT
from ROOT import TMVA, TFile, TString, TTree, TLorentzVector, TChain
from array import array
from subprocess import call
from os.path import isfile
from ROOT import std
import array
random.seed()

m=28
n=28
Matrix = [[0 for x in range(n)] for y in range(m)] 


fout = ROOT.TFile("track.root","RECREATE")
t = TTree( 'trk', 'trk' )


vecttab=[]
for x in range (n*m):
	vecttab.append(array.array('i', [0]))
	
ss=[]
for x in range (n):
	for y in range (m):
		ss.append("a["+str(x)+"]"+"["+str(y)+"]")
		t.Branch(ss[28*x+y],vecttab[28*x+y], ss[28*x+y]+"/I")
t.Fill()

for i in range (20):
	a=random.uniform(-3,3)
	b=random.uniform(0,28)
	ctr=0
	Matrix = [[0 for x in range(n)] for y in range(m)] 
	for x in range (n):
		y=(n-b-x)/a
		y=int(y)
		if (y<m and y>=0):
			Matrix[x][y]="*"
			ctr+=1
	if ctr>5:
		for x in range (n):
			print Matrix[x]
		print "-----------------"
	else:
		i-=1
		
	'''for x in range (28):
		for y in range (28):
			vecttab[28*x+y][0]=Matrix[x][y]
			#print Matrix[x][y], vecttab[28*x+y][0]
	t.Fill()
'''
t.Write()
fout.Write()
fout.Close()