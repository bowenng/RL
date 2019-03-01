import torch
from torch import nn
from torch import optim
import gym
import numpy as np
import cv2

import collections
import time

from tensorboardX import SummaryWriter


class ProcessFrameEnv(gym.ObservationWrapper):
    """
    Changes the frames to gray scale
    Down-samples the frames to (120, 84, 1)
    Center cut the image to (84, 84, 1)

    Change the observation_space to Box of dimension(84, 84, 1)
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1))

    def observation(self, obs):
        # gray scale
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # down sample
        obs = cv2.resize(obs, (120, 84), interpolation=cv2.INTER_AREA)
        # center cut
        obs = obs[12: 102, :]
        # add color channel dimension
        obs = np.resize(obs, (84, 84, 1))
        return obs


class MaxAndSkipFrameEnv(gym.Wrapper):
    """
    Applies the same action skip_frame times
    Accumulates the total rewards
    Overrides Step to return the max pixels of the last two frames
    """

    def __init__(self, env, skip_frames=4):
        super().__init__(env)
        self.skip_frames = skip_frames
        # To keep the last 2 frames
        self.last_two_frames = collections.deque(maxlen=2)

    def step(self, action):
        accumulated_rewards = 0.0
        # repeat actions skip_frames times
        for _ in range(self.skip_frames):
            next_obs, reward, done, info = self.env.step(action)
            accumulated_rewards += reward
            self.last_two_frames.append(next_obs)
            if done:
                break
        # stack up the last frames
        last_two_frames_stacked = np.stack(self.last_two_frames)
        # get the max of the last 2 frames
        new_obs = np.max(last_two_frames_stacked, axis=0)

        return new_obs, accumulated_rewards, done, info

    def reset(self):
        self.last_two_frames.clear()
        obs = self.env.reset()
        self.last_two_frames.append(obs)
        return obs


class ConvertToPytorchDimensionEnv(gym.ObservationWrapper):
    """
    Convert observation from HWC to CHW

    Change the dimension of observation_space from HWC to CHW
    """

    def __init__(self, env):
        super().__init__(env)
        # change axis for oberservation_space
        old_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=(old_shape[-1], old_shape[0], old_shape[1]))

    def observation(self, obs):
        # move axis 2 to 0, i.e C to the 0 axis
        obs = np.moveaxis(obs, 2, 0)
        return obs


class NormalizeFrameEnv(gym.ObservationWrapper):
    """
    Normalize image pixel to be of range 0.0 - 1.0
    """

    def __init__(self, env):
        super().__init__(env)
        # change range for observation_space
        old_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=old_shape, dtype=np.float32)

    def observation(self, obs):
        obs = np.array(obs).astype(np.float32) / 255.0
        return obs


class StackFramesEnv(gym.ObservationWrapper):
    """
    Stack n frames together
    """

    def __init__(self, env, stack_n):
        super().__init__(env)
        self.frames = np.zeros_like(self.observation_space.sample().repeat(stack_n, axis=0))
        self.observation_space = gym.spaces.Box(low=0, high=1.0, shape=self.frames.shape)

    def observation(self, obs):
        # discard the oldest frame
        self.frames[:-1] = self.frames[1:]
        self.frames[-1] = obs
        return self.frames

    def reset(self):
        self.frames = np.zeros_like(self.observation_space.low)
        return self.observation(self.env.reset())


class DQN(nn.Module):
    def __init__(self, input_size, n_actions):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(input_size[0], 32, kernel_size=8, stride=4),
                                  nn.ReLU(),
                                  nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                  nn.ReLU())

        conv_out_size = self.get_conv_output_size(input_size)

        self.fc = nn.Sequential(nn.Linear(conv_out_size, 512),
                                nn.ReLU(),
                                nn.Linear(512, n_actions))

    def get_conv_output_size(self, input_size):
        fake_input = torch.zeros(1, *input_size)
        return int(np.prod(self.conv(fake_input).size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceBuffer(object):
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        index = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, new_states = zip(*(np.array(self.buffer[i]) for i in index))

        return states, actions, rewards, dones, new_states


class Agent(object):
    def __init__(self, env, exp_buffer, DQN, epsilon=1.0, final_epsilon=0.02, epsilon_decay_rate=1e-4, device='cpu'):
        self.env = env
        self.exp_buffer = exp_buffer
        self.device = device
        self.DQN = DQN.to(self.device)
        self.total_reward = 0
        self.obs = None
        self.reset()
        self.epsilon = epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate


    def reset(self):
        self.obs = env.reset()
        self.total_reward = 0.0

    def play_step(self):
        play_random = np.random.random() < self.epsilon

        if play_random:
            action = self.env.action_space.sample()
        else:
            obs = torch.FloatTensor(np.array([self.obs], copy=False)).to(self.device)
            Q_values = self.DQN(obs).data.numpy()
            action = np.argmax(Q_values)

        new_obs, reward, done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.obs, action, reward, done, new_obs)
        self.exp_buffer.append(exp)

        self.obs = new_obs

        episode_reward = None

        if done:
            episode_reward = self.total_reward
            self.reset()

        self.epsilon = max(self.epsilon - self.epsilon_decay_rate, self.final_epsilon)

        return episode_reward


class Trainer(object):
    def __init__(self, agent, loss, optimizer, batch_size=32, tgt_update=1000, device='cpu'):
        self.agent = agent
        self.DQN = agent.DQN.to(device)
        self.tgt_DQN = DQN(agent.env.observation_space.shape, agent.env.action_space.n).to(device)
        self.loss = loss
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.device = device
        self.tgt_DQN_update_schedule = tgt_update
        self.steps = 0

    def train(self):
        self.optimizer.zero_grad()
        states, actions, rewards, dones, tgt_states = self.agent.exp_buffer.sample(self.batch_size)
        states_tensor = torch.tensor(states).to(self.device)
        actions_tensor = torch.tensor(actions).to(self.device)
        rewards_tensor = torch.tensor(rewards).to(self.device)
        done_mask = torch.ByteTensor(dones).to(self.device)
        tgt_states_tensor = torch.tensor(new_states).to(self.device)

        Q_values = self.DQN(states_tensor).gather(1, actions_tensor.unsqueeze(-1).squeeze(-1))
        tgt_Q_values = self.tgt_DQN(tgt_states_tensor).max(1)[0]
        tgt_Q_values[done_mask] = 0.0
        tgt_Q_values = tgt_Q_values.detach()

        expected_Q_values = rewards_tensor + tgt_Q_values * GAMMA

        L = self.loss(Q_values, expected_Q_values)
        L.backward()
        self.optimizer.step()
        self.steps += 1

        if self.steps % self.tgt_DQN_update_schedule == 0:
            self.tgt_DQN.load_state_dict(self.DQN.state_dict())

if __name__ == '__main__':
    #define constants
    DEVICE = 'cuda'
    GAMMA = 0.99
    BATCH_SIZE = 32
    REPLAY_SIZE = 10000
    REPLAY_START_SIZE = 10000
    LEARN_RATE = 1e-4
    SYNC_TARGET_FRAMES = 1000

    EPSILON_DECAY_LAST_FRAME = 10**5
    EPSILON_START = 1.0
    EPSILON_FINAL = 0.02

    ENV_NAME = 'Pong-v0'

    REWARD_BOUND = 19

    env = gym.make(ENV_NAME)
    env = MaxAndSkipFrameEnv(env)
    env = ProcessFrameEnv(env)
    env = ConvertToPytorchDimensionEnv(env)
    env = StackFramesEnv(env, 4)
    env = NormalizeFrameEnv(env)

    net = DQN(env.observation_space.shape, env.action_space.n)

    exp_buffer = ExperienceBuffer(REPLAY_SIZE)

    agent = Agent(DQN=net, env=env, exp_buffer=exp_buffer)

    trainer = Trainer(agent=agent, loss=nn.MSELoss(), optimizer=optim.Adam(agent.DQN.parameters()))
    writer = SummaryWriter(comment='-' + ENV_NAME)

    ts = time.time()
    ts_frame = 0
    frame_idx = 0
    best_mean_reward = None
    episode_rewards = []

    while True:
        frame_idx += 1
        reward = agent.play_step()
        if reward is not None:
            episode_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_reward = np.mean(episode_rewards[-100:])
            print('Played {} games; mean reward={:.3f}; speed={:.3f}'.format(len(episode_rewards), mean_reward, speed))
            writer.add_scalar('mean_reward', mean_reward, frame_idx)
            writer.add_scalar('reward', reward, frame_idx)

            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(net.state_dict(), ENV_NAME + '-best.dat')
                if best_mean_reward is not None:
                    print('Best mean reward updated: {:.3f} -> {:.3f}'.format(best_mean_reward, mean_reward))
                best_mean_reward = mean_reward
                if mean_reward > REWARD_BOUND:
                    print('Done in {} games! Best mean reward = {}!'.format(len(episode_rewards), best_mean_reward))