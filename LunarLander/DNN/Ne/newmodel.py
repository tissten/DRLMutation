from modelB import ActorCritic
import torch
import gym
import random
import os
from gym.envs.box2d import LunarLander
from PIL import Image
import numpy as np
from gym import Wrapper
from gym.envs.registration import register

def tst(n_episodes=500, name='LunarLander_ORIGIN_origin_WU1.pth'):


    # 然后就可以像使用标准环境一样使用你的自定义环境了
    env = gym.make('CustomEnv-v0')

    policy = ActorCritic()

    policy.load_state_dict(torch.load('./preTrained/{}'.format(name)))
    # 打印模型的 state_dict
    print("Model's state_dict:")
    for param_tensor in policy.state_dict():
        print(param_tensor, "\t", policy.state_dict()[param_tensor].size())



    # 假设 policy 是你的模型实例

    # 打印 affine 层的权重和偏置
    print("Affine Layer Weights:\n", policy.affine.weight.data)
    print("Affine Layer Biases:\n", policy.affine.bias.data)

    # 或者直接访问特定层的权重
    # 例如第一个卷积层的权重
    num_rows, num_cols = policy.affine.weight.data.size()
    print(num_rows, num_cols)
    # 遍历每一列
    for row in range(num_rows):
        print(policy.affine.weight.data[row])
    print("g")

    for row in range(num_rows):
        if row % 5 == 0:
            policy.affine.weight.data[row]=random.uniform(-1,1)
    for row in range(num_rows):
        print(policy.affine.weight.data[row])

    if not os.path.exists('preTrained'):
        os.makedirs('preTrained')
    torch.save(policy.state_dict(), './preTrained/LunarLander_{}_{}_{}.pth'.format("NE", "affine", "DISO_T"))



if __name__ == '__main__':
    tst()
