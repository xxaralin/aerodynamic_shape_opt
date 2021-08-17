from os import times
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F 
from tensorboardX import SummaryWriter

import gym
#import roboschool
import sys
import datcom_gym_env
#env = gym.make('Datcom-v1')

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)
#networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action,transfer = False):
        super(Actor,self).__init__()

        self.l1=nn.Linear(state_dim,400)
        self.l2=nn.Linear(400,300)
        self.l3=nn.Linear(300,action_dim)
        self.max_action=max_action

    def forward(self,x):
        x=F.relu(self.l1(x))
        x=F.relu(self.l2(x))
        x=self.max_action = np.array([(1.75-1.25)/self.action_size , (3.2-3.0)/self.action_size ,
                                     (0.4-0.1)/self.action_size , (0.09-0.0)/self.action_size ,
                                      (0.4-0.1)/self.action_size , (0.25-0.0)/self.action_size ,
                                       (0.3-0.1)/self.action_size , (0.3-0.1)/self.action_size],
                        dtype=np.float32)
        return x
        
class Critic(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(Critic,self).__init__()

        #Q1
        self.l1=nn.Liner(state_dim + action_dim,400)
        self.l2=nn.Linear(400,300)
        self.l3=nn.Linear(300,1)

        #Q2
        self.l4=nn.Liner(state_dim + action_dim,400)
        self.l5=nn.Linear(400,300)
        self.l6=nn.Linear(300,1)

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
    def __init__(self,max_size=1000000):
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
    def __init__(self, state_dim, action_dim, max_action,env):
        self.actor=Actor(state_dim,action_dim, max_action).to(device)
        self.actor_target=Actor(state_dim,action_dim,max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=1e-3)

        self.critic=Critic(state_dim,action_dim).to(device)
        self.critic_target=Critic(state_dim,action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=1e-3)

        self.max_action=max_action
        self.env=env

    def select_action(self, state,noise=0.1):#pretty self explanatory
        state=torch.FloatTensor(state.reshape(1,-1)).to(device)
        print(state)

        action=self.actor(state).cpu().data.numpy().flatten()
        if noise!=0:
            action=(action+np.random.normal(0, noise, size=self.env.action_space.shape[0]))

        return action.clip(self.env.action_space.low, self.env.action_space.high)

    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99,tau=0.005,policy_noise=0.2,noise_clip=0.5, policy_freq=2):
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
                actor_loss=+self.critic.Q1(state, self.actor(state)).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backwards()
                self.actor_optimizer.step()

    def save(self,filename, directory):
        torch.save(self.actor.state_dict(), '%s%s_actor.pth'% (directory, filename))
        torch.save(self.critic.state_dict(), '%s%s_critic.pth'% (directory, filename))

    def load(self,filename="datcom_env.py", directory="./Desktop/staj/Datcom-Env/datcom_gym_env/envs"): ####hangi belge yükleniyor???
        self.actor.load_state_dict(torch.load('%s%s_actor.pth'%(directory, filename)))
        self.critic.load_state_dict(torch.load('%s%s_critic.pth'%(directory, filename)))

class Runner():
    def __init__(self, env,agent,replay_buffer):
        self.env=env
        self.agent=agent
        self.replay_buffer=replay_buffer
        self.obs=env.reset()
        self.done=False

    def next_step(self,episode_timesteps, noise=0.1):
        action=self.agent.select_action(np.array(self.obs),noise=0.1)
        normed_state=(state-env.state_lower)/(env.state_upper-env.state_lower)
        new_obs, reward, done, _=self.env.step(action,normed_state)
        done_bool=0 if episode_timesteps+1==200 else float (done)

        replay_buffer.add((self.obs, new_obs, action, reward, done_bool))
        self.obs=new_obs
        if done:
            self.obs=self.env.reset()
            done=False

            return reward, True
        return reward, done

#evaluate
def evaluate_policy(policy, env, eval_episodes=100, render=False):
    avg_reward=0
    for i in range(eval_episodes):
        obs=env.reset()
        done=False
        while not done:
            if render:
                env.render()
            action=policy.select_action(np.array(obs),noise=0)
            normed_state=(state-env.state_lower)/(env.state_upper-env.state_lower)
            obs, reward, done, _=env.step(action,normed_state)
            avg_reward+=reward
    avg_reward/=eval_episodes

    print("\n---------------------------------------")
    print("Evaluation over {:d} episodes: {:f}" .format(eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward

    #observation-runs episodes fills buffer
def observe(env, replay_buffer, observation_steps):
    time_steps=0
    obs=env.reset()
    done=False

    while time_steps< observation_steps:
        action=env.action_space.sample()
        normed_state=(state-env.state_lower)/(env.state_upper-env.state_lower)
        new_obs, reward, done, _=env.step(action,normed_state)

        replay_buffer.add((obs,new_obs,action,done))

        obs=new_obs
        time_steps+=1

        if done:
            obs=env.reset()
            done=False
        print("\rPopulating Buffer {}/{}.".format(time_steps, observation_steps), end="")
        sys.stdout.flush()

def train(agent,test_env):#train for exploration
    total_timesteps=0
    timesteps_since_eval=0
    episode_num=0
    episode_reward=0
    episode_timesteps=0
    done=False 
    obs = env.reset()
    evaluations=[]
    rewards=[]
    best_avg=-2000

    writer=SummaryWriter(comment="-TD3_Baseline_HalfCheetah")

    while total_timesteps<EXPLORATION:
        if done:
            if total_timesteps!=0:
                rewards.append(episode_reward)
                avg_reward=np.mean(rewards[-100:])

                writer.add_scalar("avg_reward", avg_reward, total_timesteps)
                writer.add_scalar("reward_steps", reward, total_timesteps)
                writer.add_scalar("episode_reward", episode_reward, total_timesteps)

                if best_avg<avg_reward:
                    best_avg=avg_reward
                    print("saving best model....\n")
                    agent.save("best_avg","saves")
                print("\rTotal T: {:d} Episode Num: {:d} Reward: {:f} Avg Reward: {:f}".format(
                    total_timesteps, episode_num, episode_reward, avg_reward), end="")
                sys.stdout.flush()

                if avg_reward>=REWARD_THRESH:
                    break
                agent.train(replay_buffer, episode_timesteps, BATCH_SIZE, GAMMA, TAU,NOISE,NOISE_CLIP, POLICY_FREQUENCY)

                if timesteps_since_eval>=EVAL_FREQUENCY:
                    timesteps_since_eval%EVAL_FREQUENCY
                    eval_reward=evaluate_policy(agent, test_env)
                    evaluations.append(avg_reward)
                    writer.add_scalar("eval_reward", eval_reward,total_timesteps)

                    if best_avg<eval_reward:
                        best_avg=eval_reward
                        print("saving best model....\n")
                        agent.save("best_avg","saves")

                episode_reward=0
                episode_timesteps=0
                episode_num+=1
        reward, done=runner.next_step(episode_timesteps) 
        episode_reward+=reward  

        episode_timesteps+=1
        total_timesteps+=1
        timesteps_since_eval+=1




ENV = "Datcom-v1"#"RoboschoolHalfCheetah-v1"
SEED = 0
OBSERVATION = 10000
EXPLORATION = 5000000
BATCH_SIZE = 100
GAMMA = 0.99
TAU = 0.005
NOISE = 0.2
NOISE_CLIP = 0.5
EXPLORE_NOISE = 0.1
POLICY_FREQUENCY = 2
EVAL_FREQUENCY = 5000
REWARD_THRESH = 8000

env = gym.make(ENV)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seeds
env.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0] 
max_action = float(env.action_space.high[0])

policy = TD3(state_dim, action_dim, max_action, env)

replay_buffer = ReplayBuffer()

runner = Runner(env, policy, replay_buffer)

total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0
done = True

observe(env, replay_buffer, OBSERVATION)
train(policy, env)


# Load trained policy
policy.load()

# watch the trained agent run 
for i in range(10):
    evaluate_policy(policy, env, render=True)

env.close()
