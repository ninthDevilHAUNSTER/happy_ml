from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Make data
a, b, c = 5.0, 25.0, 7.0
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = a * np.outer(np.cos(u), np.sin(v))
y = b * np.outer(np.sin(u), np.sin(v))
z = c * np.outer(np.ones(np.size(u)), np.cos(v))

print(x, y, z)



# Plot the surface
ax.plot_surface(x, y, z, color='b', cmap=cm.coolwarm)

cset = ax.contourf(x, y, z, zdir='x', offset=-2 * a, cmap=cm.coolwarm)
cset = ax.contourf(x, y, z, zdir='y', offset=1.8 * b, cmap=cm.coolwarm)
cset = ax.contourf(x, y, z, zdir='z', offset=-2 * c, cmap=cm.coolwarm)

ax.set_xlabel('X')
ax.set_xlim(-2 * a, 2 * a)
ax.set_ylabel('Y')
ax.set_ylim(-1.8 * b, 1.8 * b)
ax.set_zlabel('Z')
ax.set_zlim(-2 * c, 2 * c)

plt.show()
