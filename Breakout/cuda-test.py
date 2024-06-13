import torch

print(torch.cuda.is_available() ) # cuda是否可用
print(torch.version.cuda)  # cuda版本

print(torch.backends.cudnn.is_available())  # cudnn是否可用
print(torch.backends.cudnn.version())
