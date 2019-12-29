import numpy as np
import matplotlib.pyplot as plt
import scipy, scipy.stats

print("started plotting")

#file1 = open("outMVPC.txt")
#file2 = open("outMMPC.txt")
file1 = open("outMVCluster.txt")
file2 = open("outMMCluster1.txt")

X1, Z1 = [],[]
for line in file1:
  l = [int(s) for s in line.split()]
  X1.append(l[4:])
  Z1.append(l[0:5])

X2, Z2 = [],[]
for line in file2:
  l = [int(s) for s in line.split()]
  X2.append(l[4:])
  Z2.append(l[0:5])

X1 = np.array(X1)
X2 = np.array(X2)
Z1 = np.array(Z1)
Z2 = np.array(Z2)

fig,ax = plt.subplots()
ax.grid()
ax.set(xlabel='x: Number of threads', ylabel='y: Time, microseconds')
fig.suptitle('Matrix x Vector with An, Am, Bm = ' + str(Z1[0][0:3]))
ax.plot( np.ndarray.flatten(Z1)[3::5],X1)
ax.legend([ " row-wise matrix x vector",
	    " row-wise matrix x vector Parallel",
	    " row-wise matrix x vector CopyDataToThread Parallel",
	    " column-wise matrix x vector",
	    " column-wise matrix x vector Parallel",
	    " column-wise matrix x vector CopyDataToThread Parallel",
	    " block-wise matrix x vector",
	    " block-wise matrix x vector Parallel",
	    " block-wise matrix x vector CopyDataToThread Parallel",])
plt.show()
#---------------------------------------
fig,ax = plt.subplots()
ax.grid()
ax.set(xlabel='x: Number of threads', ylabel='y: Time, microseconds')
fig.suptitle('Matrix x Matrix with An, Am, Bm = ' + str(Z2[0][0:3]))
ax.plot( np.ndarray.flatten(Z2)[3::5],X2)
ax.legend([ " basic matrix x matrix",
	    " basic matrix x matrix Parallel",
	    " column-wise (on B) matrix x matrix",
	    " column-wise (on B) matrix x matrix Parallel",
	    " column-wise (on B) matrix x matrix CopyDataToThread Parallel",
	    " block-wise matrix x matrix",
	    " block-wise matrix x matrix Parallel",
	    " block-wise matrix x matrix CopyDataToThread Parallel"])
plt.show()
