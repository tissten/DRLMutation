import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
import os
import random

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.affine = nn.Linear(8, 128)
        
        self.action_layer = nn.Linear(128, 4)
        self.value_layer = nn.Linear(128, 1)
        
        self.logprobs = []
        self.state_values = []
        self.rewards = []


    def forward(self, state,flag = False):
        state = torch.from_numpy(state).float()
        state = F.relu(self.affine(state))

        state_value = self.value_layer(state)

        action_probs = F.softmax(self.action_layer(state),dim=-1)
        if flag:

            all_probs = 1
            modified_action_probs = torch.zeros_like(action_probs)

            modified_action_probs[0] = random.uniform(0, all_probs)
            all_probs -= modified_action_probs[0]

            modified_action_probs[1] = random.uniform(0, all_probs)
            all_probs -= modified_action_probs[1]

            modified_action_probs[2] = random.uniform(0, all_probs)
            all_probs -= modified_action_probs[2]

            modified_action_probs[3] = all_probs

            action_probs = modified_action_probs  # 更新 action_probs


        action_distribution = Categorical(action_probs)
        self.state_values.append(state_value)

        
        return action_distribution
    def selectaction(self,action_distribution,flag):
        if flag:
            action = action_distribution.sample()
            self.logprobs.append(action_distribution.log_prob(action))
        else:
            action = action_distribution.sample()
            self.logprobs.append(action_distribution.log_prob(action))
        return action.item()

    def calculateLoss(self, gamma):
        # calculating discounted rewards:
        rewards = []
        dis_reward = 0
        for reward in self.rewards[::-1]:
            dis_reward = reward + gamma * dis_reward
            rewards.insert(0, dis_reward)

        # normalizing the rewards:
        rewards = torch.tensor(rewards) # Added dtype to ensure it's float32
        rewards_normalized = (rewards - rewards.mean()) / (
                    rewards.std())  # Added a small value to avoid division by zero

        loss = 0
        for logprob, value, reward in zip(self.logprobs, self.state_values, rewards_normalized):
            advantage = reward - value.item()
            action_loss = (-logprob * advantage).sum()  # Ensure it's a scalar
            reward_tensor = reward.unsqueeze(-1).to(value.device)  # Convert reward to the same shape as value
            value_loss = F.smooth_l1_loss(value, reward_tensor).sum()  # Ensure it's a scalar
            loss += (action_loss + value_loss)
        return loss

    def clearMemory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]

    def outs(self,episode):

        plt.plot(np.array(self.mm))
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join(["mstate", str(episode)])))