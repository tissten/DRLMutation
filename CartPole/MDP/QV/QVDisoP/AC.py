"""
Actor-Critic 
-------------
It uses TD-error as the Advantage.
Actor Critic History
----------------------
A3C > DDPG > AC
Advantage
----------
AC converge faster than Policy Gradient.
Disadvantage (IMPORTANT)
------------------------
The Policy is oscillated (difficult to converge), DDPG can solve
this problem using advantage of DQN.
Reference
----------
paper: https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf
View more on MorvanZhou's tutorial page: https://morvanzhou.github.io/tutorials/
Environment
------------
CartPole-v0: https://gym.openai.com/envs/CartPole-v0
A pole is attached by an un-actuated joint to a cart, which moves along a
frictionless track. The system is controlled by applying a force of +1 or -1
to the cart. The pendulum starts upright, and the goal is to prevent it from
falling over.
A reward of +1 is provided for every timestep that the pole remains upright.
The episode ends when the pole is more than 15 degrees from vertical, or the
cart moves more than 2.4 units from the center.
Prerequisites
--------------
tensorflow >=2.0.0a0
tensorlayer >=2.0.0
To run
------
python tutorial_AC.py --train/test
"""
import argparse
import time
import matplotlib.pyplot as plt
import os
import random

import gym
import numpy as np
import tensorflow as tf

import tensorlayer as tl

tl.logging.set_verbosity(tl.logging.DEBUG)

# add arguments in command  --train/test
parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=False)
args = parser.parse_args()

#####################  hyper parameters  ####################
#CartPole-v1是正常情况 CartPole-v2是状态重复 CartPole-v3是状态突变 CartPole-v4是引入状态高斯噪声
ENV_ID = 'CartPole-v1'  # environment id
RANDOM_SEED = 2  # random seed, can be either an int number or None
RENDER = False  # render while training

ALG_NAME = 'AC'
TRAIN_EPISODES = 2000  # number of overall episodes for training
TEST_EPISODES = 10  # number of overall episodes for testing
MAX_STEPS = 500  # maximum time step in one episode
LAM = 0.9  # reward discount in TD error
LR_A = 0.001  # learning rate for actor
LR_C = 0.01  # learning rate for critic



###############################  Actor-Critic  ####################################


class Actor(object):

    def __init__(self, state_dim, action_num, lr=0.001):

        input_layer = tl.layers.Input([None, state_dim], name='state')
        layer = tl.layers.Dense(
            n_units=30, act=tf.nn.relu6, W_init=tf.random_uniform_initializer(0, 0.01), name='hidden'
        )(input_layer)
        layer = tl.layers.Dense(n_units=action_num, name='actions')(layer)
        self.model = tl.models.Model(inputs=input_layer, outputs=layer, name="Actor")

        self.model.train()
        self.optimizer = tf.optimizers.Adam(lr)

    def learn(self, state, action, td_error):
        with tf.GradientTape() as tape:
            _logits = self.model(np.array([state]))
            ## cross-entropy loss weighted by td-error (advantage),
            # the cross-entropy mearsures the difference of two probability distributions: the predicted logits and sampled action distribution,
            # then weighted by the td-error: small difference of real and predict actions for large td-error (advantage); and vice versa.
            _exp_v = tl.rein.cross_entropy_reward_loss(logits=_logits, actions=[action], rewards=td_error[0])
        grad = tape.gradient(_exp_v, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_weights))
        return _exp_v

    def get_action(self, state, greedy=False):
        _logits = self.model(np.array([state]))
        _probs = tf.nn.softmax(_logits).numpy()
        if greedy:
            return np.argmax(_probs.ravel())
        return tl.rein.choice_action_by_probs(_probs.ravel())  # sample according to probability distribution

    def save(self):  # save trained weights
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        if not os.path.exists(path):
            os.makedirs(path)
        tl.files.save_npz(self.model.trainable_weights, name=os.path.join(path, 'model_actor.npz'))

    def load(self):  # load trained weights
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        tl.files.load_and_assign_npz(name=os.path.join(path, 'model_actor.npz'), network=self.model)


class Critic(object):

    def __init__(self, state_dim, lr=0.01):
        input_layer = tl.layers.Input([1, state_dim], name='state')
        layer = tl.layers.Dense(
            n_units=30, act=tf.nn.relu6, W_init=tf.random_uniform_initializer(0, 0.01), name='hidden'
        )(input_layer)
        layer = tl.layers.Dense(n_units=1, act=None, name='value')(layer)
        self.model = tl.models.Model(inputs=input_layer, outputs=layer, name="Critic")
        self.model.train()

        self.optimizer = tf.optimizers.Adam(lr)

    def learn(self, state, reward, state_, flag, done):
        d = 0 if done else 1
        v_ = self.model(np.array([state_]))
        
        with tf.GradientTape() as tape:
            v = self.model(np.array([state]))
            # print("before v", v)
            
            ## TD_error = r + d * lambda * V(newS) - V(S)
            if flag % 10 == 0:
                v += random.uniform(-1, 1)
            # print("after v", v.numpy()[0][0])

            td_error = reward + d * LAM * v_ - v
            loss = tf.square(td_error)
        grad = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_weights))
        return td_error

    def save(self):  # save trained weights
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        if not os.path.exists(path):
            os.makedirs(path)
        tl.files.save_npz(self.model.trainable_weights, name=os.path.join(path, 'model_critic.npz'))

    def load(self):  # load trained weights
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        tl.files.load_and_assign_npz(name=os.path.join(path, 'model_critic.npz'), network=self.model)


if __name__ == '__main__':
    ''' 
    choose environment
    1. Openai gym:
    env = gym.make()
    2. DeepMind Control Suite:
    env = dm_control2gym.make()
    '''
    env = gym.make(ENV_ID).unwrapped
    # dm_control2gym.create_render_mode('example mode', show=True, return_pixel=False, height=240, width=320, camera_id=-1, overlays=(),
    #              depth=False, scene_option=None)
    # env = dm_control2gym.make(domain_name="cartpole", task_name="balance")

    env.seed(RANDOM_SEED)  # reproducible
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)  # reproducible
 
    N_F = env.observation_space.shape[0]
    N_A = env.action_space.n

    print("observation dimension: %d" % N_F)  # 4
    print("observation high: %s" % env.observation_space.high)  # [ 2.4 , inf , 0.41887902 , inf]
    print("observation low : %s" % env.observation_space.low)  # [-2.4 , -inf , -0.41887902 , -inf]
    print("num of actions: %d" % N_A)  # 2 : left or right

    actor = Actor(state_dim=N_F, action_num=N_A, lr=LR_A)
    # we need a good teacher, so the teacher should learn faster than the actor
    critic = Critic(state_dim=N_F, lr=LR_C)
    finish = 0
    t0 = time.time()
    if args.train:
        all_episode_reward = []
        i = 0
        #all_state_vel = []
        for episode in range(TRAIN_EPISODES):
            state = env.reset().astype(np.float32)
            step = 0  # number of step in this episode
            episode_reward = 0  # rewards of all steps
            flag = 1
            
            while True:
                if RENDER: env.render()

                action = actor.get_action(state)

                state_new, reward, done, info = env.step(action)
                state_new = state_new.astype(np.float32)


                if done: reward = -20   # reward shaping trick
                # these may helpful in some tasks
                # if abs(s_new[0]) >= env.observation_space.high[0]:
                # #  cart moves more than 2.4 units from the center
                #     r = -20
                # reward for the distance between cart to the center
                # r -= abs(s_new[0])  * .1

                episode_reward += reward

                try:
                    td_error = critic.learn(
                        state, reward, state_new, flag, done
                    )  # learn Value-function : gradient = grad[r + lambda * V(s_new) - V(s)]
                    actor.learn(state, action, td_error)  # learn Policy : true_gradient = grad[logPi(s, a) * td_error]
                except KeyboardInterrupt:  # if Ctrl+C at running actor.learn(), then save model, or exit if not at actor.learn()
                    actor.save()
                    critic.save()

                state = state_new
                #print(state)
                step += 1
                flag += 1

                #统计速度值
                #all_state_vel.append(state[3])

                if done or step >= MAX_STEPS:
                    break
            
            # i += 1
            # print("i:", i)
            # if i > 153:
            #     print("jump out")
            #     break
            if episode == 0:
                all_episode_reward.append(episode_reward)
            else:
                all_episode_reward.append(all_episode_reward[-1] * 0.9 + episode_reward * 0.1)
            

            print('Training  | Episode: {}/{}  | Episode Reward: {:.0f}  | Running Time: {:.4f}' \
                  .format(episode + 1, TRAIN_EPISODES, episode_reward, time.time() - t0))

            # Early Stopping for quick check
            if step >= MAX_STEPS:
                print("Early Stopping")     # Hao Dong: it is important for this task
                break
        actor.save()
        critic.save()
        
        np.savetxt("QvDisoP training.txt", all_episode_reward, fmt="%.14f")
        plt.plot(all_episode_reward)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join([ALG_NAME, ENV_ID])))

       # print('state value:', np.max(all_state_vel), np.min(all_state_vel))
        #画出所有的状态集
        #plt.plot(all_state_vel)
        #plt.savefig(os.path.join('image', '_'.join([ALG_NAME, ENV_ID, 'all_state_vel'])))

    if args.test:
        actor.load()
        critic.load()

        for episode in range(TEST_EPISODES):
            episode_time = time.time()
            state = env.reset().astype(np.float32)
            t = 0  # number of step in this episode
            episode_reward = 0
            while True:
                env.render()
                action = actor.get_action(state, greedy=True)
                state_new, reward, done, info = env.step(action)
                state_new = state_new.astype(np.float32)
                if done: reward = -20

                episode_reward += reward
                state = state_new
                t += 1

                if done or t >= MAX_STEPS:
                    print('Testing  | Episode: {}/{}  | Episode Reward: {:.0f}  | Running Time: {:.4f}' \
                          .format(episode + 1, TEST_EPISODES, episode_reward, time.time() - t0))
                    break