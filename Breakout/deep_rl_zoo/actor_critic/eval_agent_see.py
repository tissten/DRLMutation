# Copyright 2022 The Deep RL Zoo Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests trained Actor-Critic agent from checkpoint with a policy-greedy actor.
on classic control tasks like CartPole, MountainCar, or LunarLander, and on Atari."""
from absl import app
from absl import flags
from absl import logging
import torch
import numpy as np

# pylint: disable=import-error
from deep_rl_zoo.networks.policy import ActorCriticMlpNet, ActorCriticConvNet
from deep_rl_zoo import main_loop
from deep_rl_zoo.checkpoint import PyTorchCheckpoint
#from deep_rl_zoo import gym_env
from deep_rl_zoo import greedy_actors
from deep_rl_zoo import gym_env_mutation_ce


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'environment_name',
    'CartPole-v1',
    'Both Classic control tasks name like CartPole-v1, LunarLander-v2, MountainCar-v0, Acrobot-v1. and Atari game like Pong, Breakout.',
)
flags.DEFINE_integer('environment_height', 84, 'Environment frame screen height, for atari only.')
flags.DEFINE_integer('environment_width', 84, 'Environment frame screen width, for atari only.')
flags.DEFINE_integer('environment_frame_skip', 4, 'Number of frames to skip, for atari only.')
flags.DEFINE_integer('environment_frame_stack', 4, 'Number of frames to stack, for atari only.')
flags.DEFINE_integer('num_iterations', 1, 'Number of evaluation iterations to run.')
flags.DEFINE_integer(
    'num_eval_steps', int(2e4), 'Number of evaluation steps (environment steps or frames) to run per iteration.'
)
flags.DEFINE_integer('max_episode_steps', 58000, 'Maximum steps (before frame skip) per episode, for atari only.')
flags.DEFINE_integer('seed', 1, 'Runtime seed.')
flags.DEFINE_bool('use_tensorboard', True, 'Use Tensorboard to monitor statistics, default on.')
flags.DEFINE_string('load_checkpoint_file', '', 'Load a specific checkpoint file.')
flags.DEFINE_string(
    'recording_video_dir',
    'recordings',
    'Path for recording a video of agent self-play.',
)
flags.DEFINE_string('mutation_object','state','muatation_object')
flags.DEFINE_string('mutation_dimensions','P','muatation_object')




def main(argv):
    """Tests ACTOR-CRITIC agent."""
    del argv
    runtime_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    random_state = np.random.RandomState(FLAGS.seed)  # pylint: disable=no-member

    # Create evaluation environments
    if FLAGS.environment_name in gym_env_mutation_ce.CLASSIC_ENV_NAMES:
        eval_env = gym_env_mutation_ce.create_classic_environment(env_name=FLAGS.environment_name, seed=random_state.randint(1, 2**10))
        state_dim = eval_env.observation_space.shape[0]
        action_dim = eval_env.action_space.n
        policy_network = ActorCriticMlpNet(state_dim=state_dim, action_dim=action_dim)
    else:
        eval_env = gym_env_mutation_ce.create_atari_environment(
            env_name=FLAGS.environment_name,
            frame_height=FLAGS.environment_height,
            frame_width=FLAGS.environment_width,
            frame_skip=FLAGS.environment_frame_skip,
            frame_stack=FLAGS.environment_frame_stack,
            max_episode_steps=FLAGS.max_episode_steps,
            seed=random_state.randint(1, 2**10),
            noop_max=30,
            terminal_on_life_loss=False,
            clip_reward=False,
            mutation_object=FLAGS.mutation_object,
            mutation_dimension=FLAGS.mutation_dimensions
        )
        state_dim = (FLAGS.environment_frame_stack, FLAGS.environment_height, FLAGS.environment_width)
        action_dim = eval_env.action_space.n
        policy_network = ActorCriticConvNet(state_dim=state_dim, action_dim=action_dim)

    logging.info('Environment: %s', FLAGS.environment_name)
    logging.info('Action spec: %s', action_dim)
    logging.info('Observation spec: %s', state_dim)

    # Setup checkpoint and load model weights from checkpoint.
    checkpoint = PyTorchCheckpoint(environment_name=FLAGS.environment_name, agent_name='Actor-Critic', restore_only=True)
    checkpoint.register_pair(('policy_network', policy_network))

    if FLAGS.load_checkpoint_file:
        checkpoint.restore(FLAGS.load_checkpoint_file)

    policy_network.eval()

    # Create evaluation agent instance
    eval_agent = greedy_actors.PolicyGreedyActor(
        network=policy_network,
        device=runtime_device,
        name='Actor-Critic',
    )

    # Run test N iterations.
    main_loop.run_evaluation_iterations(
        num_iterations=FLAGS.num_iterations,
        num_eval_steps=FLAGS.num_eval_steps,
        eval_agent=eval_agent,
        eval_env=eval_env,
        use_tensorboard=FLAGS.use_tensorboard,
        recording_video_dir=FLAGS.recording_video_dir,
    )


if __name__ == '__main__':
    app.run(main)
