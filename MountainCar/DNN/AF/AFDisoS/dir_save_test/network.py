import torch

model = torch.load('we0_actor_final.pth')



print(model['2.weight'])
print(model['2.bias'])