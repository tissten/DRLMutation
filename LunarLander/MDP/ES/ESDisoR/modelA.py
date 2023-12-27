import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
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
        self.action_counts = [0]*5
        self.total_counts = 0

    def forward(self, state):
        state = torch.from_numpy(state).float()
        state = F.relu(self.affine(state))
        
        state_value = self.value_layer(state)
        
        action_probs = F.softmax(self.action_layer(state))
        action_distribution = Categorical(action_probs)
        self.state_values.append(state_value)
        
        return action_distribution
    def selectaction(self,action_distribution,flag):
        if flag:
            action = action_distribution.sample()
            self.total_counts+=1
            self.action_counts[action]+=1
            self.logprobs.append(action_distribution.log_prob(action))
        else:
            action_value = []

            action_probs = action_distribution.probs.detach().numpy()

            for i in range(4):
                action_value.append(action_probs[i]+np.sqrt((np.log(self.total_counts+1))/(self.action_counts[i]+1)))

            action = np.argmax(action_value)

            self.total_counts+=1
            self.action_counts[action]+=1
            action = torch.tensor([action], dtype=torch.long)
            self.logprobs.append(action_distribution.log_prob(action))
        return action.item()


    def calculateLoss(self, gamma=0.99):
        
        # calculating discounted rewards:
        rewards = []
        dis_reward = 0
        for reward in self.rewards[::-1]:
            dis_reward = reward + gamma * dis_reward
            rewards.insert(0, dis_reward)
                
        # normalizing the rewards:
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std())
        
        loss = 0
        for logprob, value, reward in zip(self.logprobs, self.state_values, rewards):
            advantage = reward  - value.item()
            action_loss = -logprob * advantage
            value_loss = F.smooth_l1_loss(value, reward)
            action_loss = action_loss.unsqueeze(0) if action_loss.dim() == 0 else action_loss
            value_loss = value_loss.unsqueeze(0) if value_loss.dim() == 0 else value_loss
            loss += (action_loss + value_loss)
        return loss
    
    def clearMemory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]
