from os import times
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F 
from tensorboardX import SummaryWriter

import gym
import sys
import datcom_gym_env
#env = gym.make('Datcom-v1')
     

args = {'max_episode': 1000,
        'log_interval': 1}

#networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action,transfer = False):
        super(Actor,self).__init__()
        self.action_size = 5

        self.l1=nn.Linear(state_dim,200)
        self.l2=nn.Linear(200,200)
        self.l3=nn.Linear(200,action_dim)
        self.max_action=max_action

    def forward(self,x):
        x=F.relu(self.l1(x))
        x=F.relu(self.l2(x))
        x=F.tanh(self.l3(x))*self.max_action

        return x
        
class Critic(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(Critic,self).__init__()

        #Q1
        self.l1=nn.Linear(state_dim + action_dim,200)
        self.l2=nn.Linear(200,200)
        self.l3=nn.Linear(200,1)

        #Q2
        self.l4=nn.Linear(state_dim + action_dim,200)
        self.l5=nn.Linear(200,200)
        self.l6=nn.Linear(200,1)

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)

        x2 = F.relu(self.l4(xu))
        x2 = F.relu(self.l5(x2))
        x2 = self.l6(x2)
        return x1, x2


    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(self.l1(xu))
        x1 = F.relu(self.l2(x1))
        x1 = self.l3(x1)
        return x1
#memory part
class ReplayBuffer(object):#holds SARS state,action,reward,next state
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
            s,a,s_,r,d=self.storage[i]
            states.append(np.array(s,copy=False))
            actions.append(np.array(a,copy=False))
            next_states.append(np.array(s_,copy=False))
            rewards.append(np.array(r,copy=False))
            dones.append(np.array(d,copy=False))
        return np.array(states),np.array(actions),np.array(next_states),np.array(rewards).reshape(-1,1),np.array(dones).reshape(-1,1)
#agent (TD3)- trains networks ve outputs actions
class TD3(object):
    def __init__(self, state_dim, action_dim, max_action,env, lr):
        self.actor=Actor(state_dim,action_dim, max_action).to(device)
        self.actor_target=Actor(state_dim,action_dim,max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=lr)

        self.critic=Critic(state_dim,action_dim).to(device)
        self.critic_target=Critic(state_dim,action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=lr)

        self.max_action=max_action
        self.env=env

    def select_action(self, state):#pretty self explanatory
        state=torch.FloatTensor(state.reshape(1,-1)).to(device)

        action=self.actor(state)
        action=action.cpu().data.numpy().flatten()


        return action.clip(self.env.action_space.low, self.env.action_space.high)

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99,tau=0.005,policy_noise=2,noise_clip=0.05, policy_freq=2):
    #trains and updates actor and critic
        for it in range(iterations):
            #sample relay buffer
            x, y, u, r, d=replay_buffer.sample(batch_size)
            state=torch.FloatTensor(x).to(device)
            action=torch.FloatTensor(y).to(device)
            next_state=torch.FloatTensor(u).to(device)
            reward=torch.FloatTensor(r).to(device)
            done=torch.FloatTensor(d).to(device)

            #select action and add noise
            noise=torch.FloatTensor(u).data.normal_(0, policy_noise).to(device)
            noise=noise.clamp(-noise_clip,noise_clip)
            next_action=(self.actor_target(next_state)+noise).clamp(-self.max_action, self.max_action)

            #target Q value
            target_Q1, target_Q2=self.critic_target(next_state, next_action)
            target_Q=torch.min(target_Q1,target_Q2)
            target_Q=reward+(done*discount*target_Q).detach()

            #current Q estimate
            current_Q1, current_Q2=self.critic(state, action)

            #critic loss
            critic_loss=F.mse_loss(current_Q1,target_Q)+F.mse_loss(current_Q2,target_Q)

            #optimize critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            #delay
            if it % policy_freq==0:
                actor_loss= -self.critic.Q1(state, self.actor(state)).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

    def save(self,filename, directory):
        torch.save(self.actor.state_dict(), '%s%s_actor.pth'% (directory, filename))
        torch.save(self.critic.state_dict(), '%s%s_critic.pth'% (directory, filename))

    def load(self,filename="deneme2", directory="./"): 
        self.actor.load_state_dict(torch.load('%s%s_actor.pth'%(directory, filename)))
        self.critic.load_state_dict(torch.load('%s%s_critic.pth'%(directory, filename)))


    #observation-runs episodes fills buffer
def observe(env, replay_buffer, observation_steps):
    time_steps=0
    obs=env.reset()
    obs=(obs-env.state_lower)/(env.state_upper-env.state_lower)
    done=False

    while time_steps< observation_steps:
        action=env.action_space.sample()
        new_obs, reward, done, _=env.step(action,obs,bıdık)
        new_obs=(new_obs-env.state_lower)/(env.state_upper-env.state_lower)
        replay_buffer.add((obs,new_obs,action, reward, done))
        obs=new_obs
        time_steps+=1

        if done:
            obs=env.reset()
            done=False
        print("\rPopulating Buffer {}/{}.".format(time_steps, observation_steps), end="")
        sys.stdout.flush()

def train(agent,noise_param=0.01, noise_clip_param=0.005, lr=1e-3):#train for exploration

    done=False 
    writer=SummaryWriter(comment=f"-noise={noise_param}-lr={lr}-reward_bıdık{bıdık}-")
    total_step = 0
    epoch = 0
    

    for i in range(args['max_episode']):
        total_reward = 0
        step =0        
        state = env.reset()
        state = (state - env.state_lower ) / (env.state_upper-env.state_lower)
     
        
        for t in range(25):   ##while(1)            
            action = agent.select_action(state)               
            #action = action * (env.state_upper-env.state_lower) +  env.state_lower
            writer.add_scalar('Main/action_0', action[0], global_step=epoch)
            writer.add_scalar('Main/action_1', action[1], global_step=epoch)
            writer.add_scalar('Main/action_2', action[2], global_step=epoch)
            writer.add_scalar('Main/action_3', action[3], global_step=epoch)
            writer.add_scalar('Main/action_4', action[4], global_step=epoch)
            writer.add_scalar('Main/action_5', action[5], global_step=epoch)
            writer.add_scalar('Main/action_6', action[6], global_step=epoch)
            writer.add_scalar('Main/action_7', action[7], global_step=epoch)
            writer.add_scalar('States/normed_state_0', state[0], global_step=epoch)
            writer.add_scalar('States/normed_state_1', state[1], global_step=epoch)
            writer.add_scalar('States/normed_state_2', state[2], global_step=epoch)
            writer.add_scalar('States/normed_state_3', state[3], global_step=epoch)
            writer.add_scalar('States/normed_state_4', state[4], global_step=epoch)
            writer.add_scalar('States/normed_state_5', state[5], global_step=epoch)
            writer.add_scalar('States/normed_state_6', state[6], global_step=epoch)
            writer.add_scalar('States/normed_state_7', state[7], global_step=epoch)
            """
            if(noise_bool==True):
                noise=0.025*2.5066*noise_param*np.random.normal(0, noise_param, size=env.action_space.shape[0])
            else:
               """
            noise = np.random.normal(0, noise_param, size=env.action_space.shape[0])
            
            

            writer.add_scalar('Main/noise_0', noise[0], global_step=epoch)
            writer.add_scalar('Main/noise_1', noise[1], global_step=epoch)
            writer.add_scalar('Main/noise_2', noise[2], global_step=epoch)
            writer.add_scalar('Main/noise_3', noise[3], global_step=epoch)
            writer.add_scalar('Main/noise_4', noise[4], global_step=epoch)
            writer.add_scalar('Main/noise_5', noise[5], global_step=epoch)
            writer.add_scalar('Main/noise_6', noise[6], global_step=epoch)
            writer.add_scalar('Main/noise_7', noise[7], global_step=epoch)

            action = action + noise 
            next_state, reward, done, info = env.step(action,state,bıdık)
            next_state = (next_state - env.state_lower ) / (env.state_upper-env.state_lower)
            replay_buffer.add((state, next_state, action, reward, np.float(done)))

            state = next_state
            

            step += 1
            total_reward += reward
            epoch += 1
            writer.add_scalar('Main/step_reward', reward, global_step=epoch)
            writer.add_scalar('Main/CL_CD', env.cl_cd, global_step=epoch)
            writer.add_scalar('States/XLE1', env.XLE1, global_step=epoch)
            writer.add_scalar('States/XLE2', env.XLE2, global_step=epoch)
            writer.add_scalar('States/CHORD1_1', env.CHORD1_1, global_step=epoch)
            writer.add_scalar('States/CHORD1_2', env.CHORD1_2, global_step=epoch)
            writer.add_scalar('States/CHORD2_1', env.CHORD2_1, global_step=epoch)
            writer.add_scalar('States/CHORD2_2', env.CHORD2_2, global_step=epoch)
            writer.add_scalar('States/SSPAN1_2', env.SSPAN1_2, global_step=epoch)
            writer.add_scalar('States/SSPAN2_2', env.SSPAN2_2, global_step=epoch)
            #writer.add_scalar('debug_reward/cl_cd_diff', env.cl_cd_diff, global_step=epoch)
            #writer.add_scalar('debug_reward/xcp_diff', env.xcp_diff, global_step=epoch)
            #writer.add_scalar('debug_reward/cd_diff', env.cd_diff, global_step=epoch)
            #writer.add_scalar('debug_reward/which', env.which, global_step=epoch)

            if done or step > 20:
                break


        total_step += step+1
        print("Total T:{} Episode: \t{} Total Reward: \t{:0.2f}".format(total_step, i, total_reward))

        agent.train(replay_buffer, step, BATCH_SIZE, GAMMA, TAU,noise_param,noise_clip_param, POLICY_FREQUENCY)
        writer.add_scalar('Main/episode_reward', total_reward, global_step=i)
        writer.add_scalar('Main/episode_steps', step, global_step=i)
        writer.add_scalar('Main/episode_CL_CD', env.cl_cd, global_step=i)
       # "Total T: %d Episode Num: %d Episode T: %d Reward: %f

        if i % args['log_interval'] == 0:
            agent.save('deneme2', './')



ENV = "Datcom-v1"
SEED = 0
OBSERVATION = 1000
BATCH_SIZE = 64
GAMMA = 0.99
EXPLORE_NOISE = 0.1
for bıdık in range(41,101,10):
    for NOISE in [1e-2,5e-2,1e-1]:
        for TAU in [5e-3]:
            for POLICY_FREQUENCY in [2]:
                for lr in [3e-3, 3e-4]:
                    for noise_bool in[True]:

                        NOISE_CLIP = 2*NOISE
                        env = gym.make(ENV)
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        torch.autograd.set_detect_anomaly(True)
                        # Set seeds
                        env.seed(SEED)
                        torch.manual_seed(SEED)
                        np.random.seed(SEED)

                        state_dim = env.observation_space.shape[0]
                        action_dim = env.action_space.shape[0] 
                        max_action = float(env.action_space.high[0])

                        policy = TD3(state_dim, action_dim, max_action, env, lr)

                        replay_buffer = ReplayBuffer()

                        total_timesteps = 0
                        timesteps_since_eval = 0
                        episode_num = 0
                        done = True

                        observe(env, replay_buffer, OBSERVATION)
                        train(policy, NOISE, NOISE_CLIP, lr)

                        policy.load()
                        
                        env.close()
