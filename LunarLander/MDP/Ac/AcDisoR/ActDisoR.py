from test import test
from modelA import ActorCritic
import torch
import torch.optim as optim
import gym
import numpy as np
import matplotlib.pyplot as plt
import os
import random


def train():
    # Defaults parameters:
    #    gamma = 0.99
    #    lr = 0.02
    #    betas = (0.9, 0.999)
    #    random_seed = 543
    ALG_NAME = 'Act'
    ENV_ID = 'Diso'
    PINLV_ID = 'P'
    render = False
    gamma = 0.99
    lr = 0.02
    betas = (0.9, 0.999)
    random_seed = 543

    torch.manual_seed(random_seed)

    env = gym.make('LunarLander-v2')
    env.seed(random_seed)

    policy = ActorCritic()
    optimizer = optim.Adam(policy.parameters(), lr=lr, betas=betas)
    print(lr, betas)
    all_episode_reward = []
    running_reward = 0
    a_episodes = 0
    for i_episode in range(0, 10000):
        flag = 0
        state = env.reset()
        episode_reward = 0
        for t in range(10000):
            if random.randint(1,100) > 95:
                action = policy(state)
                action = policy.selectaction(action,False)
            else:
                action = policy(state)
                action = policy.selectaction(action,True)
            state, reward, done, _ = env.step(action)
            policy.rewards.append(reward)
            running_reward += reward
            episode_reward += reward
            flag+=1
            if render and i_episode > 1000:
                env.render()
            if done:
                break

        # Updating the policy :
        optimizer.zero_grad()
        loss = policy.calculateLoss(gamma)
        loss.backward()
        optimizer.step()
        policy.clearMemory()

        # saving the model if episodes > 999 OR avg reward > 200
        # if i_episode > 999:
        #    torch.save(policy.state_dict(), './preTrained/LunarLander_{}_{}_{}.pth'.format(lr, betas[0], betas[1]))
        if episode_reward > 199:
            a_episodes+=1
        if episode_reward < 200:
            a_episodes = 0
        if i_episode == 0:
            all_episode_reward.append(episode_reward)
        else:
            all_episode_reward.append(all_episode_reward[-1] * 0.9 + episode_reward * 0.1)
        if a_episodes > 30:
            if not os.path.exists('preTrained'):
                os.makedirs('preTrained')
            torch.save(policy.state_dict(), './preTrained/LunarLander_{}_{}_{}.pth'.format(ALG_NAME, ENV_ID, PINLV_ID))
            print("########## Solved! ##########")
            # test(name='LunarLander_{}_{}_{}.pth'.format(lr, betas[0], betas[1]))
            np.savetxt("ActDisoP training.txt", all_episode_reward, fmt="%.14f")  # 记录
            plt.plot(all_episode_reward)
            if not os.path.exists('image'):
                os.makedirs('image')
            plt.savefig(os.path.join('image', '_'.join([ALG_NAME, ENV_ID, PINLV_ID])))
            break

        if i_episode % 20 == 0:
            running_reward = running_reward / 20
            print('Episode {}\tlength: {}\treward: {}'.format(i_episode, t, running_reward))
            running_reward = 0
        if i_episode ==20000:
            if not os.path.exists('preTrained'):
                os.makedirs('preTrained')
            torch.save(policy.state_dict(), './preTrained/LunarLander_{}_{}_{}.pth'.format("No_solved", ENV_ID, PINLV_ID))
            print("########## Solved! ##########")
            # test(name='LunarLander_{}_{}_{}.pth'.format(lr, betas[0], betas[1]))
            np.savetxt("ActDisoP No solved.txt", all_episode_reward, fmt="%.14f")  # 记录
            plt.plot(all_episode_reward)
            if not os.path.exists('image'):
                os.makedirs('image')
            plt.savefig(os.path.join('image', '_'.join(["NO_solved", ENV_ID, PINLV_ID])))
            break


if __name__ == '__main__':
    train()
