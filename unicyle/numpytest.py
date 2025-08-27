import numpy as np

# a = np.array([[1,2,3],[4,5,6]])
# b = np.array([[7,8,9],[10,11,12]])

# # x = np.vstack((a,b))
# # print(x)
# # y = np.hstack((a,b))
# # print(y)
# z = np.dstack((a,b))
# print(z[:, :, 0])

state_diff = np.array([0, 0, 3.141592653589793 +3.078760800517997])
while abs(state_diff[2]) >= np.pi:
    state_diff[2]+= np.pi * -np.sign(state_diff[2])

print(state_diff)