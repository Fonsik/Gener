import random
from termcolor import colored

random.seed()

m=28
n=28
Matrix = [[0 for x in range(n)] for y in range(m)] 


a=random.uniform(-1,1)
b=random.uniform(0,28)

for x in range (28):
	y=a*x+b
	y=int(y)
	if (y<28and y>=0):
		Matrix[x][y]="*"

for x in range (n):
	print Matrix[x]