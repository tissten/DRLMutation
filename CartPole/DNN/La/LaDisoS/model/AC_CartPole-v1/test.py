

import numpy as np
import os
import random
import tensorflow as tf

data = np.load("model_actor.npz")

data1 = np.load("test.npz")
print(data1['params'])



# w1 = data['params'][0]
# b1 = data['params'][1]

w1 = []
b1 = []

for i in range(len(data['params'][0])):
    w = []
    for j in range(len(data['params'][0][0])):
        w.append(random.uniform(-0.3, 0.3))
    w1.append(w)
for i in range(len(data['params'][1])):
    b1.append(random.uniform(-0.3, 0.3))

print(len(b1))

# print("w1 is:", w1)
# print("b1 is:", b1)
w1 = np.array(w1)
b1 = np.array(b1)

w2 = data['params'][2]
b2 = data['params'][3]
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
