import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F 
from tensorboardX import SummaryWriter

import gym
import roboschool
import sys
import datcom_gym_env
#env = gym.make('Datcom-v1')



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
    def
    