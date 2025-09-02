import numpy as np

x = (np.pi-0.1 + np.pi) % (2 * np.pi) - np.pi
y = (-np.pi+0.1 + np.pi) % (2 * np.pi) - np.pi
b = x-y
b = (b + np.pi) % (2 * np.pi) - np.pi
print(b)