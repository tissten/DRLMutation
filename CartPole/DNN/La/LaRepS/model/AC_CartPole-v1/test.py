

import numpy as np
import os

data = np.load("model_actor.npz")


w1 = data['params'][0]
b1 = data['params'][1]
w2 = data['params'][2]
b2 = data['params'][3]
w3 = data['params'][2]
b3 = data['params'][3]
w4 = data['params'][4]
b4 = data['params'][5]
# data['params'][0][1].pop(0)

# w = np.dot(data['params'][0], data['params'][2])

# b = np.dot(data['params'][1], data['params'][2]) + data['params'][3]

# print("w is:", w)
# print("b is:", b)

# print([w, b])

np.savez("test.npz", params = [w1, b1, w2, b2, w3, b3, w4, b4])


# test = np.load("test.npz")
# print("is ", test.files)
# print(data['params'])
