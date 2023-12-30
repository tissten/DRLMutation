import torch
import random
import numpy as np
from collections import OrderedDict

model = torch.load('we0_actor_final.pth')



# print(model['2.weight'])
# print(model['2.bias'])

w_o_0 = model['0.weight'].numpy()
b_o_0 = model['0.bias'].numpy()

print(w_o_0.shape[1])
print(b_o_0.shape[0])
w1 = []
# b1 = []

for i in range(w_o_0.shape[0]):
    w = []
    for j in range(w_o_0.shape[1]):
        w.append(np.float32(format(w_o_0[i][j], '.2f')))
    w1.append(w)
# for i in range(b_o_0.shape[0]):
#     b1.append(random.uniform(-0.3, 0.3))



w1 = torch.tensor(w1)
b1 = model['0.bias']

w2 = model['2.weight']
b2 = model['2.bias']



ordered_dict = OrderedDict()
ordered_dict['0.weight'] = w1
ordered_dict['0.bias'] = b1
ordered_dict['2.weight'] = w2
ordered_dict['2.bias'] = b2

print(ordered_dict)

torch.save(ordered_dict, 'we0_actor_final.pth')
# print(model)
# print(b1)