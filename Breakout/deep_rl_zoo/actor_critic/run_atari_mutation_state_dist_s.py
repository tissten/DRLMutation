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
"""From the paper "Actor-Critic Algorithms"
https://proceedings.neurips.cc/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf.
"""
from absl import app
from absl import flags
from absl import logging
import numpy as np
import torch

# pylint: disable=import-error
from deep_rl_zoo.networks.policy import ActorCriticConvNet
from deep_rl_zoo.actor_critic import agent_gamma_d_p
from deep_rl_zoo.actor_critic import agent
from deep_rl_zoo.checkpoint import PyTorchCheckpoint
from deep_rl_zoo import main_loop
#from deep_rl_zoo import gym_env
from deep_rl_zoo import greedy_actors
from deep_rl_zoo import replay as replay_lib
from deep_rl_zoo import gym_env_state_di_s

FLAGS = flags.FLAGS
flags.DEFINE_string('environment_name', 'Pong', 'Atari name without NoFrameskip and version, like Breakout, Pong, Seaquest.')
flags.DEFINE_integer('environment_height', 84, 'Environment frame screen height.')
flags.DEFINE_integer('environment_width', 84, 'Environment frame screen width.')
flags.DEFINE_integer('environment_frame_skip', 4, 'Number of frames to skip.')
flags.DEFINE_integer('environment_frame_stack', 4, 'Number of frames to stack.')
flags.DEFINE_bool('clip_grad', False, 'Clip gradients, default off.')
flags.DEFINE_float('max_grad_norm', 40.0, 'Max gradients norm when do gradients clip.')
flags.DEFINE_float('learning_rate', 0.0005, 'Learning rate.')
flags.DEFINE_float('discount', 0.99, 'Discount rate.')
flags.DEFINE_float('entropy_coef', 0.05, 'Coefficient for the entropy loss.')
flags.DEFINE_float('value_coef', 0.5, 'Coefficient for the state-value loss.')
flags.DEFINE_integer('batch_size', 32, 'Accumulate batch size transitions before do learning.')
flags.DEFINE_integer('num_iterations', 100, 'Number of iterations to run.')
flags.DEFINE_integer(
    'num_train_steps', int(6e5), 'Number of training steps (environment steps or frames) to run per iteration.'
)
flags.DEFINE_integer(
    'num_eval_steps', int(2e4), 'Number of evaluation steps (environment steps or frames) to run per iteration.'
)
flags.DEFINE_integer('max_episode_steps', 108000, 'Maximum steps (before frame skip) per episode.')
flags.DEFINE_integer('seed', 1, 'Runtime seed.')
flags.DEFINE_bool('use_tensorboard', True, 'Use Tensorboard to monitor statistics, default on.')
flags.DEFINE_bool('actors_on_gpu', True, 'Run actors on GPU, default on.')
flags.DEFINE_integer(
    'debug_screenshots_interval',
    0,
    'Take screenshots every N episodes and log to Tensorboard, default 0 no screenshots.',
)
flags.DEFINE_string('tag', '_state_di_s_f', 'Add tag to Tensorboard log file.')
flags.DEFINE_string('results_csv_path', './logs/actor_critic_action_p.csv', 'Path for CSV log file.')
flags.DEFINE_string('checkpoint_dir', './state', 'Path for checkpoint directory.')
flags.DEFINE_string('mutation_object','state','mutation_object')
flags.DEFINE_string('mutation_dimensions','_di_s_f','mutation_object')
flags.DEFINE_string('load_checkpoint_file', '', 'Load a specific checkpoint file.')



def main(argv):
    """Trains Actor-Critic agent on Atari."""
    del argv
    runtime_device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Runs Actor-Critic agent on {runtime_device}')
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    random_state = np.random.RandomState(FLAGS.seed)  # pylint: disable=no-member

    # Create environment.
    def environment_builder():
        return gym_env_state_di_s.create_atari_environment(
            env_name=FLAGS.environment_name,
            frame_height=FLAGS.environment_height,
            frame_width=FLAGS.environment_width,
            frame_skip=FLAGS.environment_frame_skip,
            frame_stack=FLAGS.environment_frame_stack,
            max_episode_steps=FLAGS.max_episode_steps,
            seed=random_state.randint(1, 2**10),
            noop_max=30,
            terminal_on_life_loss=True,
            mutation_object= FLAGS.mutation_object,
            mutation_dimension =FLAGS.mutation_dimensions
        )

    train_env = environment_builder()
    eval_env = environment_builder()

    logging.info('Environment: %s', FLAGS.environment_name)
    logging.info('Action spec: %s', train_env.action_space.n)
    logging.info('Observation spec: %s', train_env.observation_space.shape)

    state_dim = train_env.observation_space.shape
    action_dim = train_env.action_space.n

    # Test environment and state shape.
    obs = train_env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == state_dim == (FLAGS.environment_frame_stack, FLAGS.environment_height, FLAGS.environment_width)

    # Create policy network and optimizer
    policy_network = ActorCriticConvNet(state_dim=state_dim, action_dim=action_dim)
    policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr=FLAGS.learning_rate)

    # Test network output.
    s = torch.from_numpy(obs[None, ...]).float()
    network_output = policy_network(s)
    assert network_output.pi_logits.shape == (1, action_dim)
    assert network_output.value.shape == (1, 1)

    # Create Actor-Critic agent instance
    train_agent = agent.ActorCritic(
        policy_network=policy_network,
        policy_optimizer=policy_optimizer,
        transition_accumulator=replay_lib.TransitionAccumulator(),
        discount=FLAGS.discount,
        batch_size=FLAGS.batch_size,
        entropy_coef=FLAGS.entropy_coef,
        value_coef=FLAGS.value_coef,
        clip_grad=FLAGS.clip_grad,
        max_grad_norm=FLAGS.max_grad_norm,
        device=runtime_device,
    )

    # Create evaluation agent instance
    eval_agent = greedy_actors.PolicyGreedyActor(
        network=policy_network,
        device=runtime_device,
        name='Actor-Critic',
    )
    #FLAGS.tag=FLAGS.mutation_dimensions+FLAGS.mutation_object
    # Setup checkpoint.
    a_name = 'Actor-Critic'+FLAGS.mutation_object+FLAGS.mutation_dimensions
    checkpoint = PyTorchCheckpoint(
        environment_name=FLAGS.environment_name, agent_name=a_name, save_dir=FLAGS.checkpoint_dir
    )
    checkpoint.register_pair(('policy_network', policy_network))

    if FLAGS.load_checkpoint_file:
        checkpoint.restore(FLAGS.load_checkpoint_file)

    # Run the training and evaluation for N iterations.
    main_loop.run_single_thread_training_iterations(
        num_iterations=FLAGS.num_iterations,
        num_train_steps=FLAGS.num_train_steps,
        num_eval_steps=FLAGS.num_eval_steps,
        train_agent=train_agent,
        train_env=train_env,
        eval_agent=eval_agent,
        eval_env=eval_env,
        checkpoint=checkpoint,
        csv_file=FLAGS.results_csv_path,
        use_tensorboard=FLAGS.use_tensorboard,
        tag=FLAGS.tag,
        debug_screenshots_interval=FLAGS.debug_screenshots_interval,
    )


if __name__ == '__main__':
    app.run(main)
