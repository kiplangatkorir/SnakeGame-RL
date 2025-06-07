import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class RLAgent:
    def __init__(self, state_size=14, action_size=3):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(100000)  # Larger memory
        self.gamma = 0.9     # Discount rate
        self.epsilon = 1.0   # Exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995  # Slower decay
        self.learning_rate = 0.00025  # Smaller learning rate
        self.model = DQN(state_size, 256, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)  # Added L2 regularization
        self.criterion = nn.SmoothL1Loss()  # Huber loss for more stable training
        self.batch_size = 1024  # Larger batch size
        self.target_update = 1000  # Steps between target network updates
        self.steps = 0
        
        # Target network
        self.target_model = DQN(state_size, 256, action_size)
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state_tensor)
        return torch.argmax(act_values).item()
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return 0.0
            
        minibatch = self.memory.sample(batch_size)
        
        states = torch.FloatTensor(np.array([t[0] for t in minibatch]))
        actions = torch.LongTensor(np.array([t[1] for t in minibatch]))
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch]))
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch]))
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch]))
        
        # Current Q values
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss
        loss = self.criterion(current_q.squeeze(), target)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Gradient clipping
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
