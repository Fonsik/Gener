from subprocess import call

noise=0
eta=0.8
i=0

for i in range(20):
	nazva=str(eta)+'vs'+str(noise)
	'''call ('python Regr2.py', +str(noise)+' '+str(eta))
	call(python cnn.py)
	call(python Selection_Rec.py)
	call(python Application.py)'''
	call('mkdir', " "+str(nazva))
	call('mkdir', " "+str(nazva)+'_new')
	#call('python test.py', str(noise)+' '+str(eta))
	#i+=1
	#noise+=0.03