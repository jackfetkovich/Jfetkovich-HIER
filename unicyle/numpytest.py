import numpy as np

a = np.array([[1,2,3],[4,5,6]])
b = np.array([[7,8,9],[10,11,12]])

# x = np.vstack((a,b))
# print(x)
# y = np.hstack((a,b))
# print(y)
z = np.dstack((a,b))
print(z[:, :, 0])