import random
import sys
from scipy.stats import norm
import numpy as np
import h5py
import matplotlib.pyplot as plt
from PIL import Image

n=28
m=28
s=1

s1=(sys.argv[1])
s2=(sys.argv[2])
folder=str(s2)+"vs"+str(s1)
nump=50000
np.set_printoptions(linewidth=90)
f=open("parametry.txt", "a")
paramp=np.load("cnn_out.npy")
paramr=np.load("data_param.npy")


ap=np.zeros((50000))
ar=np.zeros((50000))
bp=np.zeros((50000))
br=np.zeros((50000))
for i in range(50000):
	ap[i]=paramp[i][0][0]
	ar[i]=paramr[i][0][0]
	bp[i]=paramp[i][0][1]
	br[i]=paramr[i][0][1]


x=ap-ar

plt.hist(x, 99)
mu, std=norm.fit(x)
print mu
print std
f.write(str(mu))
f.write(str(std))

plt.savefig("./"+folder+"/Hist_a_conv.png", dpi=300)
plt.clf()
x=bp-br

plt.hist(x, 99)
mu, std=norm.fit(x)
f.write(str(mu))
f.write(str(std))
plt.savefig("./"+folder+"/Hist_b_cov.png", dpi=300)
plt.clf()

plt.scatter(ar, ap, s)
plt.savefig("./"+folder+"/arVSapcov.png", dpi=300)
plt.clf()

plt.scatter(br, bp,s)
plt.savefig("./"+folder+"/brVSbpcov.png", dpi=300)
plt.clf()


paramp=np.load("dense_out.npy")

ap=np.zeros((50000))
bp=np.zeros((50000))
for i in range(50000):
	ap[i]=paramp[i][0][0]
	bp[i]=paramp[i][0][1]


x=ap-ar

plt.hist(x, 99)
mu, std=norm.fit(x)
f.write(str(mu))
f.write(str(std))
plt.savefig("./"+folder+"/Hist_a_dense.png", dpi=300)
plt.clf()
x=bp-br

plt.hist(x, 99)
mu, std=norm.fit(x)
f.write(str(mu))
f.write(str(std))
plt.savefig("./"+folder+"/Hist_b_dense.png", dpi=300)
plt.clf()

plt.scatter(ar, ap, s)
plt.savefig("./"+folder+"/arVSap_dense.png", dpi=300)
plt.clf()

plt.scatter(br, bp, s)
plt.savefig("./"+folder+"/brVSbp_dense.png", dpi=300)
plt.clf()