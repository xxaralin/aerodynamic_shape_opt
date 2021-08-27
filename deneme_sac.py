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
import datcom_gym_env

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

"""class ValueNetwork(nn.Module):
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
        return x"""

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
        
        std = torch.exp(log_std)
        
        log_prob = None
        
        if deterministic:
            action = torch.tanh(mean)
        else:
            # assumes actions have been normalized to (0,1)
            normal = Normal(0, 1)
            z = mean + std * normal.sample().requires_grad_()
            action = torch.tanh(mean + std * normal.sample().requires_grad_())
            log_prob = Normal(mean, std).log_prob(z) - torch.log(1 - action * action + self.epsilon)
            
        return action, mean, log_std, log_prob, std
    
    def get_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action,_,_,_,_ =  self.forward(state, deterministic)
        act = action.cpu()[0][0]
        return act.clip(torch.FloatTensor(env.action_space.low),torch.FloatTensor( env.action_space.high))
    """
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        
        mean    = self.mean_linear(x)
        
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        mean,log_std=self.forward(state)
        std =log_std.exp()

        normal = Normal(0, 1)
        z = mean + std * normal.sample().requires_grad_()
        action = torch.tanh(z)
        log_prob = Normal(mean, std).log_prob(z) - torch.log(1 - action * action + self.epsilon)
        return action, mean, log_std, log_prob, std
    
    def get_action(self, state):
        state = torch.FloatTensor(state.reshape(1,-1)).to(device)
        action,_,_,_ =  self.forward(state)
        act = action.cpu()[0][0]
        return act"""
        
class ReplayBuffer(object):
    def __init__(self,max_size=1000):
        self.storage=[]
        self.max_size=max_size
        self.ptr=0

    def add(self,data):#experience replay tuple-agents experience at a time t(yine sars)
        if len(self.storage)==self.max_size:
            self.storage[int(self.ptr)]=data
            self.ptr=(self.ptr+1)% self.max_size
            #storage dolduysa baştan tekrar doldurmaya başlıyor i guess?
        else:
            self.storage.append(data)
    def sample(self,batch_size):#train edeceği batch size
        ind=np.random.randint(0,len(self.storage),size=batch_size) #random number within the range
        states,actions,next_states,rewards,dones=[],[],[],[],[]
        for i in ind:
            s,a,r,s_,d=self.storage[i]
            states.append(np.array(s,copy=False))
            actions.append(np.array(a,copy=False))
            next_states.append(np.array(s_,copy=False))
            rewards.append(np.array(r,copy=False))
            dones.append(np.array(d,copy=False))
        return np.array(states),np.array(actions),np.array(rewards).reshape(-1,1),np.array(next_states),np.array(dones).reshape(-1,1)
  

class SAC(object):
    
    def __init__(self, env, replay_buffer, seed=0, hidden_dim=200,
        steps_per_epoch=200, epochs=1000, discount=0.99,
        tau=1e-2, lr=1e-3, auto_alpha=True, batch_size=100, start_steps=10000,
        max_ep_len=200, logger_kwargs=dict(), save_freq=1):
        
        # Set seeds
        self.env = env
        self.obs=env.reset()
        self.env.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # env space
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0] 
        self.hidden_dim = hidden_dim
        
        # device
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        
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
  
    def get_action(self, state,deterministic=False, explore=False):
        
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        if explore:
            return self.env.action_space.sample()
        else:
            action  = self.policy_net.get_action(state,deterministic).detach()
            return action.numpy()
        
    def update(self, iterations, batch_size = 103):
        
        for _ in range(0,iterations):
        
            state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
            print(action.shape,reward.shape,state.shape,next_state.shape)

            state      = torch.FloatTensor(state).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            action     = torch.FloatTensor(action).to(device)
            reward     = torch.FloatTensor(reward).unsqueeze(1).to(device)
            done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

            new_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy_net(state)

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
            print((alpha*log_pi - q_new_actions).shape,policy_loss,q_new_actions.shape)

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
def train(agent, steps_per_epoch=100, epochs=100, start_steps=100, max_ep_len=20):

    writer=SummaryWriter(comment=f" ")
    
    # start tracking time
    start_time = time.time()
    
    obs, r, d, ep_reward, ep_len, ep_num = env.reset(), 0, False, 0, 0, 1
    
    # track total steps
    total_steps = steps_per_epoch * epochs
    
    for t in range(0,total_steps):
        
        explore = t < start_steps
        a = agent.get_action(obs,True,explore)
        normed_state=(agent.obs-env.state_lower)/(env.state_upper-env.state_lower)
        

        writer.add_scalar('Main/action_0', a[0], global_step=t)
        writer.add_scalar('Main/action_1', a[1], global_step=t)
        writer.add_scalar('Main/action_2', a[2], global_step=t)
        writer.add_scalar('Main/action_3', a[3], global_step=t)
        writer.add_scalar('Main/action_4', a[4], global_step=t)
        writer.add_scalar('Main/action_5', a[5], global_step=t)
        writer.add_scalar('Main/action_6', a[6], global_step=t)
        writer.add_scalar('Main/action_7', a[7], global_step=t)
        writer.add_scalar('States/normed_state_0', normed_state[0], global_step=t)
        writer.add_scalar('States/normed_state_1', normed_state[1], global_step=t)
        writer.add_scalar('States/normed_state_2', normed_state[2], global_step=t)
        writer.add_scalar('States/normed_state_3', normed_state[3], global_step=t)
        writer.add_scalar('States/normed_state_4', normed_state[4], global_step=t)
        writer.add_scalar('States/normed_state_5', normed_state[5], global_step=t)
        writer.add_scalar('States/normed_state_6', normed_state[6], global_step=t)
        writer.add_scalar('States/normed_state_7', normed_state[7], global_step=t)
        
###        # Step the env
        new_obs, r, d, _ = env.step(a,normed_state)
        ep_reward += r
        ep_len += 1

        d = False if ep_len == max_ep_len else d
        replay_buffer.add((obs, a, r, new_obs, d))
        obs = new_obs
        writer.add_scalar('Main/step_reward', r, global_step=t)
        writer.add_scalar('Main/CL_CD', env.cl_cd, global_step=t)
        writer.add_scalar('States/XLE1', env.XLE1, global_step=t)
        writer.add_scalar('States/XLE2', env.XLE2, global_step=t)
        writer.add_scalar('States/CHORD1_1', env.CHORD1_1, global_step=t)
        writer.add_scalar('States/CHORD1_2', env.CHORD1_2, global_step=t)
        writer.add_scalar('States/CHORD2_1', env.CHORD2_1, global_step=t)
        writer.add_scalar('States/CHORD2_2', env.CHORD2_2, global_step=t)
        writer.add_scalar('States/SSPAN1_2', env.SSPAN1_2, global_step=t)
        writer.add_scalar('States/SSPAN2_2', env.SSPAN2_2, global_step=t)
        
        if d or (ep_len == max_ep_len):
        
            # carry out update for each step experienced (episode length)
            if not explore:
                agent.update(ep_len)
            
            print("Steps:{} Episode:{} Reward:{} ".format(t, ep_num, ep_reward))
#             logger.store(LossPi=outs[0], LossQ1=outs[1], LossQ2=outs[2],
#                          LossV=outs[3], Q1Vals=outs[4], Q2Vals=outs[5],
#                          VVals=outs[6], LogPi=outs[7])

#             logger.store(EpRet=ep_ret, EpLen=ep_len)
            obs, r, d, ep_reward, ep_len = env.reset(), 0, False, 0, 0
            ep_num += 1


replay_buffer = ReplayBuffer()

env = gym.make("Datcom-v1")
hidden_dim=200
agent = SAC(env, replay_buffer)
train(agent) 

env.close()