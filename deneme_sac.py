#SAC
import math
import random
import sys
import gym
import numpy as np
import time 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from tensorboardX import SummaryWriter
hidden_dim=200
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, 200)
        self.linear2 = nn.Linear(200, 200)
        self.linear3 = nn.Linear(200, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=[400,300], init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs + num_actions, 200)
        self.linear2 = nn.Linear(200,200)
        self.linear3 = nn.Linear(200, 1)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=[400,300], 
                 init_w=3e-3, log_std_min=-20, log_std_max=2, epsilon=1e-6):
        super(PolicyNetwork, self).__init__()
        
        self.epsilon = epsilon
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(num_inputs, 200)
        self.linear2 = nn.Linear(200,200)
        
        self.mean_linear = nn.Linear(200, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(200, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)
    
    def forward(self, state, deterministic=False):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        
        mean    = self.mean_linear(x)
        
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def evaluate(self,state,epsilon=1e-6):
        mean,log_std=self.forward(state)
        std =log_std.exp()

        normal = Normal(0, 1)
        z = mean + std * normal.sample().requires_grad_()
        action = torch.tanh(z)
        log_prob = Normal(mean, std).log_prob(z) - torch.log(1 - action * action + self.epsilon)
        return action, mean, log_std, log_prob, std
    
    def get_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action,_,_,_,_ =  self.forward(state, deterministic)
        act = action.cpu()[0][0]
        return act
        
class ReplayBuffer:
    def __init__(self, max_size=1000):
        self.max_size=max_size
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.max_size:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.max_size
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)
        
class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low  = self.action_space.low
        high = self.action_space.high

        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action

    def normalize_action(action, low, high):
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action
    
class SAC(object):
    
    def __init__(self, env, replay_buffer, seed=0, hidden_dim=200,
        steps_per_epoch=200, epochs=1000, discount=0.99,
        tau=1e-2, lr=1e-3, auto_alpha=True, batch_size=100, start_steps=10000,
        max_ep_len=200, logger_kwargs=dict(), save_freq=1):
        
        # Set seeds
        self.env = env
        self.env.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # env space
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0] 
        self.hidden_dim = hidden_dim
        
        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # init networks
        
        # Soft Q
        self.soft_q_net1 = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(device)
        self.soft_q_net2 = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(device)
        
        self.target_soft_q_net1 = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(device)
        self.target_soft_q_net2 = SoftQNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(device)
        
        for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
            target_param.data.copy_(param.data)
        
        for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
            target_param.data.copy_(param.data)
            
        # Policy
        self.policy_net = PolicyNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(device)
        
        # Optimizers/Loss
        self.soft_q_criterion = nn.MSELoss()
        
        self.soft_q_optimizer1 = optim.Adam(self.soft_q_net1.parameters(), lr=lr)
        self.soft_q_optimizer2 = optim.Adam(self.soft_q_net2.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # alpha tuning
        self.auto_alpha = auto_alpha
        
        if self.auto_alpha:
            self.target_entropy = -np.prod(env.action_space.shape).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            
        self.replay_buffer = replay_buffer
        self.discount = discount
        self.batch_size = batch_size
        self.tau = tau
        
    def get_action(self, state, deterministic=False, explore=False):
        
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        if explore:
            return self.env.action_space.sample()
        else:
            action  = self.policy_net.get_action(state, deterministic).detach()
            return action.numpy()
           
    def update(self, iterations, batch_size = 100):
        
        for _ in range(0,iterations):
        
            state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

            state      = torch.FloatTensor(state).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            action     = torch.FloatTensor(action).to(device)
            reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
            done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

            new_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy_net.evaluate(state)

            if self.auto_alpha:
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                alpha = self.log_alpha.exp()
            else:
                alpha_loss = 0
                alpha = 0.2 # constant used by OpenAI

            # Update Policy 
            q_new_actions = torch.min(
                self.soft_q_net1(state, new_actions), 
                self.soft_q_net2(state, new_actions)
            )

            policy_loss = (alpha*log_pi - q_new_actions).mean()

            # Update Soft Q Function
            q1_pred = self.soft_q_net1(state, action)
            q2_pred = self.soft_q_net2(state, action)

            new_next_actions, _, _, new_log_pi, *_ = self.policy_net(next_state)

            target_q_values = torch.min(
                self.target_soft_q_net1(next_state, new_next_actions),
                self.target_soft_q_net2(next_state, new_next_actions),
            ) - alpha * new_log_pi

            q_target = reward + (1 - done) * self.discount * target_q_values
            q1_loss = self.soft_q_criterion(q1_pred, q_target.detach())
            q2_loss = self.soft_q_criterion(q2_pred, q_target.detach())

            # Update Networks
            self.soft_q_optimizer1.zero_grad()
            q1_loss.backward()
            self.soft_q_optimizer1.step()

            self.soft_q_optimizer2.zero_grad()
            q2_loss.backward()
            self.soft_q_optimizer2.step()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            # Soft Updates
            for target_param, param in zip(self.target_soft_q_net1.parameters(), self.soft_q_net1.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.tau) + param.data * self.tau
                )

            for target_param, param in zip(self.target_soft_q_net2.parameters(), self.soft_q_net2.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.tau) + param.data * self.tau
                )