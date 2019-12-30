import numpy as np
import matplotlib.pyplot as plt
import scipy, scipy.stats

print("started plotting")


file1 = open("outEmpty.txt")
file2 = open("outContains.txt")
#file1 = open("outEmptyPC.txt")
#file2 = open("outContainsPC.txt")
#file1 = open("outEmptyCluster.txt")
#file2 = open("outContainsCluster.txt")


X1, Z1 = [],[]
for line in file1:
  l = [int(s) for s in line.split()]
  X1.append(l[3:])
  Z1.append(l[0:4])

X2, Z2 = [],[]
for line in file2:
  l = [int(s) for s in line.split()]
  X2.append(l[3:])
  Z2.append(l[0:4])

X1 = np.array(X1)
X2 = np.array(X2)
Z1 = np.array(Z1)
Z2 = np.array(Z2)

fig,ax = plt.subplots()
ax.grid()
ax.set(xlabel='x: Number of threads', ylabel='y: Time, microseconds')
fig.suptitle('Search random substring in random string (likely not constains in string) with N, m = ' + str(Z1[0][0:2]))
ax.plot( np.ndarray.flatten(Z1)[2::4],X1)
ax.legend([ " Basic search",
	    " Parallel search",
	    " CopyDataToThread Parallel search"])
plt.show()
#---------------------------------------
fig,ax = plt.subplots()
ax.grid()
ax.set(xlabel='x: Number of threads', ylabel='y: Time, microseconds')
fig.suptitle('Search substring (that contains in string) in random string with N, m = ' + str(Z2[0][0:2]))
ax.plot( np.ndarray.flatten(Z2)[2::4],X2)
ax.legend([ " Basic search",
	    " Parallel search",
	    " CopyDataToThread Parallel search"])
plt.show()
