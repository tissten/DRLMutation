import torch
import random
import numpy as np
from collections import OrderedDict

model = torch.load('we0_actor_final.pth')



# print(model['2.weight'])
# print(model['2.bias'])

w_o_0 = model['0.weight'].numpy()
b_o_0 = model['0.bias'].numpy()
w_o_1 = model['2.weight'].numpy()
b_o_1 = model['2.bias'].numpy()

print(w_o_0.shape)
print(b_o_0.shape)
print(w_o_1.shape)
print(b_o_1.shape)


w1 = []
b1 = []

# for i in range(w_o_0.shape[0]):
w_o_0[random.randint(1,53)][random.randint(0,1)] = random.uniform(-0.3, 0.3)
    # w = []
    # for j in range(w_o_0.shape[1]):
    #     w.append(random.uniform(-0.3, 0.3))
    # w1.append(w)
# for i in range(b_o_0.shape[0]):
b_o_0[random.randint(1,63)] = random.uniform(-0.3, 0.3)
    # b1.append(random.uniform(-0.3, 0.3))

# w1 = torch.tensor(w1)
# b1 = torch.tensor(b1)

w2 = model['2.weight']
b2 = model['2.bias']



ordered_dict = OrderedDict()
ordered_dict['0.weight'] = w_o_0
ordered_dict['0.bias'] = b_o_0
ordered_dict['2.weight'] = w2
ordered_dict['2.bias'] = b2




# print(ordered_dict)

torch.save(ordered_dict, 'we0_actor_final.pth')


# print(model)
# print(b1)