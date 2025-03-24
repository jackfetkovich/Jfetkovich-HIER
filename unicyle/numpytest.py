import numpy as np


K = 10
T = 4
weights = np.linspace(1, 10, 10)
print(weights.shape)

v = np.random.normal(loc=0, scale=1, size=(K, T))
w = np.random.normal(loc=0, scale=np.pi, size=(K, T))

U = np.stack([v, w], axis=-1)
print(v)
print(w)
print(U)
