import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

Num = 5000
A = [[0.54, 0.28, 1.76]]
B = [[-54, -28, -76]]
C = [[854, 7628, 176]]

A = np.tile(np.array(A), (Num, 1))
B = np.tile(np.array(B), (Num, 1))
C = np.tile(np.array(C), (Num, 1))

s = np.tile(np.expand_dims(np.random.rand(Num), 1), (1, 3))
t = np.tile(np.expand_dims(np.random.rand(Num), 1), (1, 3))


D = s * B + t * C + A
print(D.shape)

fig = plt.figure()
ax1 = plt.axes(projection='3d')

ax1.scatter3D(D[:, 0], D[:, 1], D[:, 2], cmap='Blues')
# ax1.plot3D(x,y,z,'gray')    #绘制空间曲线
plt.show()

