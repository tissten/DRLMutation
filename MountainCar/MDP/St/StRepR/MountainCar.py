import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import argparse
from  collections  import deque
import time
from model import Policy
from ppo import ppo_agent
from storage import RolloutStorage
from utils import get_render_func, get_vec_normalize
from envs import make_vec_envs
from parallelEnv import parallelEnv
import matplotlib.pyplot as plt
import os


parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=False)
args = parser.parse_args()

print('gym version: ', gym.__version__)
print('torch version: ', torch.__version__)

seed = 0 
gamma=0.99
num_processes =  20

device = torch.device("cpu")
print('device: ', device)


class Other(object):


    def save(model, directory, filename, suffix):
        torch.save(model.base.actor.state_dict(), '%s/%s_actor_%s.pth' % (directory, filename, suffix))
        torch.save(model.base.critic.state_dict(), '%s/%s_critic_%s.pth' % (directory, filename, suffix))
        torch.save(model.base.critic_linear.state_dict(), '%s/%s_critic_linear_%s.pth' % (directory, filename, suffix))
        torch.save(model.base, '%s/%s_model_base_%s.pth' % (directory, filename, suffix))
        torch.save(model.dist, '%s/%s_model_dist_%s.pth' % (directory, filename, suffix))


    def return_suffix(j, limits):
        suf = '0'
        for i in range(len(limits)-1):
            if j > limits[i] and j < limits[i+1]:
                suf = str(limits[i+1])
                break
            
            i_last = len(limits)-1    
            if  j > limits[i_last]:
                suf = str(limits[i_last])
                break
        return suf 
    
    def load_test(model):
        model.base = torch.load('dir_save_test\we0_model_base_final.pth')
        model.base.actor.load_state_dict(torch.load('dir_save_test\we0_actor_final.pth'))
        # model.base.critic.load_state_dict(torch.load('dir_save\we0_critic_final.pth'))
        # model.base.critic_linear.load_state_dict(torch.load('dir_save\we0_critic_linear_final.pth'))
        model.dist = torch.load('dir_save_test\we0_model_dist_final.pth')





if __name__ == '__main__':
    


    
    
    if args.train:

######################      Training环境加载          ######################################

        envs = parallelEnv('MountainCarContinuous-v0', n=num_processes, seed=seed)

        ## make_vec_envs -cannot find context for 'forkserver'
        ## forkserver is only available in Python 3.4+ and only on some Unix platforms (not on Windows).
        ## envs = make_vec_envs('BipedalWalker-v2', \
        ##                    seed + 1000, num_processes,
        ##                    None, None, False, device='cpu', allow_early_resets=False)

        limits = [-300, -160, -100, -70, -50, 0, 20, 30, 40, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
        num_updates=100
        gamma = 0.99
        tau=0.95
        save_interval=30
        log_interval= 1 
        num_steps = 999

        max_steps = envs.max_steps
        print('max_steps: ', max_steps)

        threshold = envs.threshold
        print('threshold: ', threshold)
        threshold = 150
        print('reassigned threshold: ', threshold)

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        dir_chk = 'dir_save_test'

        ## model Policy uses MLPBase
        policy = Policy(envs.observation_space.shape, envs.action_space,\
                base_kwargs={'recurrent': False})

        policy.to(device)

        agent = ppo_agent(actor_critic=policy, ppo_epoch=16, num_mini_batch=16,\
                        lr=0.01, eps=1e-5, max_grad_norm=0.5)

        rollouts = RolloutStorage(num_steps=max_steps, num_processes=num_processes, \
                                obs_shape=envs.observation_space.shape, action_space=envs.action_space, \
                                recurrent_hidden_state_size=policy.recurrent_hidden_state_size)

        obs = envs.reset()
        print('type obs: ', type(obs), ', shape obs: ', obs.shape)
        obs_t = torch.tensor(obs)
        print('type obs_t: ', type(obs_t), ', shape obs_t: ', obs_t.shape)

        rollouts.obs[0].copy_(obs_t)
        rollouts.to(device)

############# Training 模型训练        ##################################################

        time_start = time.time()
    
        n=len(envs.ps)    
        envs.reset()
        
        # start all parallel agents
        print('Number of agents: ', n)
        envs.step([[1]*4]*n)
        
        indices = []
        for i  in range(n):
            indices.append(i)
        
        s = 0
        
        scores_deque = deque(maxlen=100)
        scores_array = []
        avg_scores_array = []    

        for i_episode in range(num_updates):
            
            total_reward = np.zeros(n)
            timestep = 0
            
            done = False
            pre_obs = []
            
            for timestep in range(num_steps):
                
                with torch.no_grad():
                    value, actions, action_log_prob, recurrent_hidden_states = \
                    policy.act(
                            rollouts.obs[timestep],
                            rollouts.recurrent_hidden_states[timestep],
                            rollouts.masks[timestep])
                    
                    
                obs, rewards, done, _ = envs.step(actions.cpu().detach().numpy())

                if timestep != 0 and random.randint(1, 100) > 90:
                    obs = pre_obs
                pre_obs = obs
                
                
                total_reward += rewards  ## this is the list by agents
                            
                # If done then clean the history of observations.
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                obs_t = torch.tensor(obs)
                
                ## Add one dimnesion to tensor, 
                ## This is (unsqueeze(1)) solution for:
                ## RuntimeError: The expanded size of the tensor (1) must match the existing size...
                rewards_t = torch.tensor(rewards).unsqueeze(1)
                rollouts.insert(obs_t, recurrent_hidden_states, actions, action_log_prob, \
                    value, rewards_t, masks)
                                    
            avg_total_reward = np.mean(total_reward)
            scores_deque.append(avg_total_reward)
            scores_array.append(avg_total_reward)
                    
            with torch.no_grad():
                next_value = policy.get_value(rollouts.obs[-1],
                                rollouts.recurrent_hidden_states[-1],
                                rollouts.masks[-1]).detach()

            rollouts.compute_returns(next_value, gamma, tau)

            agent.update(rollouts)

            rollouts.after_update()
            
            avg_score = np.mean(scores_deque)
            avg_scores_array.append(avg_score)

            if i_episode > 0 and i_episode % save_interval == 0:
                print('Saving model, i_episode: ', i_episode, '\n')
                suf = Other.return_suffix(avg_score, limits)
                Other.save(policy, dir_chk, 'we0', suf)

            
            if i_episode % log_interval == 0 and len(scores_deque) > 1:            
                prev_s = s
                s = (int)(time.time() - time_start)
                t_del = s - prev_s
                print('Ep. {}, Timesteps {}, Score.Agents: {:.2f}, Avg.Score: {:.2f}, Time: {:02}:{:02}:{:02}, \
    Interval: {:02}:{:02}'\
                    .format(i_episode, timestep+1, \
                            avg_total_reward, avg_score, s//3600, s%3600//60, s%60, t_del%3600//60, t_del%60)) 
        
            if len(scores_deque) > 1 and avg_score > threshold:   
                print('Environment solved with Average Score: ',  avg_score)
                break
    
    
    if args.train:

##############       Training模型保存     ###############################################################
        
        Other.save(model=policy,directory=dir_chk,filename='we0',suffix='final')

        np.savetxt("StRepR training.txt", scores_array, fmt="%.14f")
        plt.plot(np.arange(1, len(scores_array)+1), scores_array, label="Score")
        plt.plot(np.arange(1, len(avg_scores_array)+1), avg_scores_array, label="Avg on 100 episodes")
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join(['AC', 'MountainCar'])))
        # plt.savefig(os.path.join('image', '_'.join(['AC', 'MountainCar'])))

        print('length of scores: ', len(scores_array), ', len of avg_scores: ', len(avg_scores_array))

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # plt.plot(np.arange(1, len(scores_array)+1), scores_array, label="Score")
        # plt.plot(np.arange(1, len(avg_scores_array)+1), avg_scores_array, label="Avg on 100 episodes")
        # plt.legend(bbox_to_anchor=(1.05, 1)) 
        # plt.ylabel('Score')
        # plt.xlabel('Episodes #')
        # plt.show()


        
    
    if args.test:

        num_episodes = 500

######################  Test 环境和模型加载   ########################################################

        device = torch.device("cpu")
        print('device: ', device)

        seed = 0 

        ## model Policy uses MLPBase
        envs = parallelEnv('MountainCarContinuous-v0', n=1, seed=seed) ## weights created by n = 16

        max_steps = envs.max_steps
        print('max_steps: ', max_steps)

        policy = Policy(envs.observation_space.shape, envs.action_space,\
                base_kwargs={'recurrent': False})

        print('policy: ', policy)
        policy.to(device)

        num_processes = 1
        env_venv = make_vec_envs('MountainCarContinuous-v0', \
                            seed + 1000, num_processes,
                            None, None, False, device=device, allow_early_resets=False)

        print('envs.observation_space.shape: ', envs.observation_space.shape, \
            ', len(obs_shape): ', len(envs.observation_space.shape))
        print('envs.action_space: ',  envs.action_space, \
            ', action_space.shape[0]: ', envs.action_space.shape[0])


        
            
        Other.load_test(model = policy)  
  

  ###################  Test模型运行   ##########################################################

        obs = env_venv.reset()
        obs = torch.Tensor(obs)
        obs = obs.float()
            
        recurrent_hidden_states = torch.zeros(1, policy.recurrent_hidden_state_size)
        
        masks = torch.zeros(1, 1)
        
        scores_deque = deque(maxlen=100)

        render_func = get_render_func(env_venv)
            
        for i_episode in range(1, num_episodes+1):     

            time_start = time.time()
            total_reward = np.zeros(num_processes)
            timestep = 0

            done = False
            
            while not done:
                # env_venv.render()
            
                with torch.no_grad():
                    value, action, _, recurrent_hidden_states = \
                        policy.act(obs, recurrent_hidden_states, masks, deterministic=False) # obs = state
                                

                # render_func()
                
                obs, reward, done, _ = env_venv.step(action.unsqueeze(1))
                obs = torch.Tensor(obs)
                obs = obs.float()

                reward = reward.detach().numpy()
                masks.fill_(0.0 if done else 1.0)
                
                total_reward += np.mean(reward)
                
                time.sleep(0.04)
                
                timestep += 1
                
                if done.all() == True or timestep + 1 == max_steps: ##   999:
                    break

            s = (int)(time.time() - time_start)
            
            scores_deque.append(total_reward)        
            avg_score = np.mean(scores_deque)
                        
            print('Episode {} \tScore: {:.2f}, Avg.Score: {:.2f}, \tTime: {:02}:{:02}:{:02}'\
                    .format(i_episode, np.mean(total_reward), avg_score,  s//3600, s%3600//60, s%60))
            
#################  Test 环境关闭 ###################################################################

        env_venv.close()
