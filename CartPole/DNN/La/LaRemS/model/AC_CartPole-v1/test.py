

import numpy as np
import os

data = np.load("model_actor.npz")

# data['params'][0][1].pop(0)

w = np.dot(data['params'][0], data['params'][2])

b = np.dot(data['params'][1], data['params'][2]) + data['params'][3]


print("w is:", w)
print("b is:", b)

print([w, b])

np.savez("test.npz", params = [w, b])


test = np.load("test.npz")
print("is ", test.files)
