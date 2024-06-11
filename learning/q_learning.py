import torch
import numpy as np
import random
from collections import deque, namedtuple

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
import copy

from learning.models import MahjongNetwork

device = torch.device("mps")
    

class ReplayBuffer:
    def __init__(self, action_size: int, buffer_size: int, batch_size: int, seed: int):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor, done: bool):
        """Add a new experience to memory."""
        self.memory.append(self.experience(state, action, reward, next_state, done))

    def sample(self) -> tuple:
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None])).bool().to(device)
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return len(self.memory)

class DQNTrainer:
    def __init__(self, model: MahjongNetwork, buffer_size: int, batch_size: int, lr: float, gamma: int, epsilon: int, seed: int):
        self.qnetwork_local = model
        torch.save(self.qnetwork_local.state_dict(), 'network.pth')
        self.qnetwork_target = self.qnetwork_local.load_state_dict(torch.load('network.pth'))
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_size = model.output_size

        self.memory = ReplayBuffer(self.action_size, buffer_size=buffer_size, batch_size=batch_size, seed=seed)

    def step(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor, done: bool):
        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) > self.memory.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma)

    def act(self, state: torch.Tensor):
        state = state.float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state).detach().cpu().numpy()
        self.qnetwork_local.train()

        #Epsilon-greedy
        best = np.argmax(action_values)
        weights = np.full_like(action_values, self.epsilon / (self.action_size - 1))
        weights[best] = 1 - self.epsilon
        return np.random.choice(self.action_size, p=weights)
    
    def act_meld(self, state: torch.Tensor):
        state = state.float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        return action_values
        #Epsilon-greedy
        best = np.argmax(action_values)
        weights = np.full_like(action_values, self.epsilon / (self.action_size - 1))
        weights[best] = 1 - self.epsilon
        return np.random.choice(self.action_size, p=weights)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        action_values = self.qnetwork_target(next_states).detach()
        max_action_values = action_values.max(1)[0].unsqueeze(1)

        q_targets = rewards + (self.gamma * max_action_values * (1 - dones.long()))
        q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target)
    
    def end_learn(self, states, rewards):
        q_targets = rewards
        q_expected = self.qnetwork_local(states).gather(1, self.memory.memory.pop().action)

        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target)
    
    def soft_update(self):
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.gamma*local_param.data + (1.0-self.gamma)*target_param.data)