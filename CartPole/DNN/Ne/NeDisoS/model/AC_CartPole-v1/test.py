

import numpy as np
import os
import random
# import tensorflow as tf

data = np.load("model_actor.npz")

# data1 = np.load("test.npz")
# print(data['params'])
w1 = data['params'][0]
print("before:", w1[0][0])
w1[0,0] = random.uniform(-0.1, 0.1)
print("after:", w1[0][0])
b1 = data['params'][1]
print("before:", b1[0])
b1[0] = random.uniform(-0.1, 0.1)
print("after:", b1[0])

w1 = data['params'][0]
b1 = data['params'][1]
w2 = data['params'][2]
b2 = data['params'][3]
# print("before:", len(w1[0]))
# w1 = np.delete(w1, 0, axis = 1)
# print("after:", len(w1[0]))

# b1 = np.delete(b1, 0)
# w2 = np.delete(w2, 0, axis = 0)

# for i in range(len(data['params'][0])):
#     np.delete()
#     data['params'][i].remove(0)
# print(data['params'][0])




# w1 = data['params'][0]
# b1 = data['params'][1]




# print("w1 is:", w1)
# print("b1 is:", b1)
# w1 = data['params'][0]

# print(len(w1[0]))
# # b1 = data['params'][1]
# print(len(b1))
# # w2 = data['params'][2]
# print(len(w2))
# b2 = data['params'][3]
# data['params'][0][1].pop(0)

# w = np.dot(data['params'][0], data['params'][2])

# b = np.dot(data['params'][1], data['params'][2]) + data['params'][3]

# print("w is:", w)
# print("b is:", b)

# print([w, b])

np.savez("test.npz", params = [w1, b1, w2, b2])


# test = np.load("test.npz")
# print("is ", test.files)
# print(data['params'])
