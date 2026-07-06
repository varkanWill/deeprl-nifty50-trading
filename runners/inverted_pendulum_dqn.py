import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
LR = 1e-3
MEMORY_SIZE = 10000
TARGET_UPDATE = 10

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START
        
        self.policy_net = DQN(state_size, action_size)
        self.target_net = DQN(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.criterion = nn.MSELoss()
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            act_values = self.policy_net(state)
        return np.argmax(act_values.numpy())
        
    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
            
        minibatch = random.sample(self.memory, BATCH_SIZE)
        states = torch.FloatTensor(np.array([t[0] for t in minibatch]))
        actions = torch.LongTensor([t[1] for t in minibatch]).unsqueeze(1)
        rewards = torch.FloatTensor([t[2] for t in minibatch])
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch]))
        dones = torch.FloatTensor([t[4] for t in minibatch])
        
        curr_Q = self.policy_net(states).gather(1, actions).squeeze(1)
        
        next_Q = self.target_net(next_states).max(1)[0]
        expected_Q = rewards + (GAMMA * next_Q * (1 - dones))
        
        loss = self.criterion(curr_Q, expected_Q.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

def train_pendulum():
    env = gym.make('CartPole-v1') # Inverted pendulum problem
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    episodes = 500
    for e in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_result
                
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            agent.replay()
            
        if agent.epsilon > EPSILON_END:
            agent.epsilon *= EPSILON_DECAY
            
        if e % TARGET_UPDATE == 0:
            agent.update_target_network()
            
        print(f"Episode {e}/{episodes} - Reward: {total_reward} - Epsilon: {agent.epsilon:.2f}")
        
        if total_reward >= 475:
            print("Environment Solved!")
            break

if __name__ == "__main__":
    train_pendulum()
