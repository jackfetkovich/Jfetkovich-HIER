import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.linspace(11, 20, 10)
z = np.linspace(21, 30, 10)

points = np.array([x, y, z])
print(np.dstack(points)[0][0])