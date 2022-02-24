import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def getSphere(a, b, c):
    x1, y1, z1 = a
    x2, y2, z2 = b
    x3, y3, z3 = c

    a1 = (y1 * z2 - y2 * z1 - y1 * z3 + y3 * z1 + y2 * z3 - y3 * z2)
    b1 = -(x1 * z2 - x2 * z1 - x1 * z3 + x3 * z1 + x2 * z3 - x3 * z2)
    c1 = (x1 * y2 - x2 * y1 - x1 * y3 + x3 * y1 + x2 * y3 - x3 * y2)
    d1 = -(x1 * y2 * z3 - x1 * y3 * z2 - x2 * y1 * z3 + x2 * y3 * z1 + x3 * y1 * z2 - x3 * y2 * z1)

    a2 = 2 * (x2 - x1)
    b2 = 2 * (y2 - y1)
    c2 = 2 * (z2 - z1)
    d2 = x1 * x1 + y1 * y1 + z1 * z1 - x2 * x2 - y2 * y2 - z2 * z2


    a3 = 2 * (x3 - x1)
    b3 = 2 * (y3 - y1)
    c3 = 2 * (z3 - z1)
    d3 = x1 * x1 + y1 * y1 + z1 * z1 - x3 * x3 - y3 * y3 - z3 * z3


    cx = -(b1 * c2 * d3 - b1 * c3 * d2 - b2 * c1 * d3 + b2 * c3 * d1 + b3 * c1 * d2 - b3 * c2 * d1) / (a1 * b2 * c3 - a1 * b3 * c2 - a2 * b1 * c3 + a2 * b3 * c1 + a3 * b1 * c2 - a3 * b2 * c1)

    cy = (a1 * c2 * d3 - a1 * c3 * d2 - a2 * c1 * d3 + a2 * c3 * d1 + a3 * c1 * d2 - a3 * c2 * d1) / (a1 * b2 * c3 - a1 * b3 * c2 - a2 * b1 * c3 + a2 * b3 * c1 + a3 * b1 * c2 - a3 * b2 * c1)

    cz = -(a1 * b2 * d3 - a1 * b3 * d2 - a2 * b1 * d3 + a2 * b3 * d1 + a3 * b1 * d2 - a3 * b2 * d1) / (a1 * b2 * c3 - a1 * b3 * c2 - a2 * b1 * c3 + a2 * b3 * c1 + a3 * b1 * c2 - a3 * b2 * c1)

    return np.array([cx, cy, cz]), np.sqrt(np.power(np.array([x1 - cx, y1 - cy, z1 - cy]), 2).sum())

Num = 5000
A = [0.54, 0.28, 1.76]
B = [-54, -28, -76]
C = [854, 7628, 176]

yuanxin, r = getSphere(A, B, C)

PI = np.pi
theta = np.linspace(0, 2 * PI, Num)

# D = r * np.sin(theta) +

u = np.linspace(0, 2 * np.pi, Num)
v = np.linspace(0, np.pi, Num)
u, v = np.meshgrid(u, v)
X = r * np.sin(v) * np.cos(u) + yuanxin[0]
Y = r * np.sin(v) * np.sin(u) + yuanxin[1]
Z = r * np.cos(v) + yuanxin[2]

fig = plt.figure()
ax1 = plt.axes(projection='3d')

ax1.scatter3D(X, Y, Z, cmap='Blues')
# ax1.plot3D(x,y,z,'gray')    #绘制空间曲线
plt.show()
