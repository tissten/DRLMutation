from model import ActorCritic
import torch
import gym
from gym.envs.box2d import LunarLander
from PIL import Image
import numpy as np
from gym import Wrapper
from gym.envs.registration import register

def tst(n_episodes=500, name='LunarLander_We_affine_Fuz_S.pth'):

    for e in range(6):
        if e == 0:
            env = gym.make('CustomEnv-v0')
        if e == 1:
            env = gym.make('CustomEnv-v1')
        if e == 2:
            env = gym.make('CustomEnv-v2')
        if e == 3:
            env = gym.make('CustomEnv-v3')
        if e == 4:
            env = gym.make('CustomEnv-v4')
        if e == 5:
            env = gym.make('CustomEnv-v5')
        policy = ActorCritic()

        policy.load_state_dict(torch.load('./preTrained/{}'.format(name)))

        render = False
        save_gif = False
        a_episodes = 0
        for i_episode in range(1, n_episodes + 1):
            state = env.reset()
            running_reward = 0
            for t in range(10000):
                action = policy(state)
                state, reward, done, _ = env.step(action)
                running_reward += reward
                if render:
                    env.render()
                    if save_gif:
                        img = env.render(mode='rgb_array')
                        img = Image.fromarray(img)
                        img.save('./gif/{}.jpg'.format(t))
                if done:
                    break
            print('Episode {}\tReward: {}'.format(i_episode, running_reward))
            if running_reward > 199:
                a_episodes += 1
        data = [a_episodes, a_episodes / 500]
        if e == 0:
            np.savetxt("test_result_y", data, fmt="%.14f")
        if e == 1:
            np.savetxt("test_result_x", data, fmt="%.14f")
        if e == 2:
            np.savetxt("test_result_x_vel", data, fmt="%.14f")
        if e == 3:
            np.savetxt("test_result_y_vel", data, fmt="%.14f")
        if e == 4:
            np.savetxt("test_result_angle", data, fmt="%.14f")
        if e == 5:
            np.savetxt("test_result_angle_vel", data, fmt="%.14f")
        env.close()
    # 然后就可以像使用标准环境一样使用你的自定义环境了




if __name__ == '__main__':
    tst()
