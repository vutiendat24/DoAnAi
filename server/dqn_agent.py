import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        c, h, w = input_shape

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(h * w, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, input_shape, n_actions):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DQN(input_shape, n_actions).to(self.device)
        self.target_model = DQN(input_shape, n_actions).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.memory = deque(maxlen=10000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.criteria = nn.MSELoss()

        self.batch_size = 64
        self.gamma = 0.99
        self.update_target_steps = 1000
        self.step_count = 0

    def select_action(self, state, epsilon, available_actions):
        if random.random() < epsilon:
            return random.choice(available_actions)
        else:
            state_v = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.model(state_v)
            q_values = q_values.cpu().data.numpy().flatten()

            q_values_invalid = np.full_like(q_values, -np.inf)
            for a in available_actions:
                q_values_invalid[a] = q_values[a]
            return int(np.argmax(q_values_invalid))

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_v = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions_v = torch.tensor(actions).to(self.device)
        rewards_v = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states_v = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones_t = torch.ByteTensor(dones).to(self.device)

        state_action_values = self.model(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        next_state_values = self.target_model(next_states_v).max(1)[0]
        next_state_values[dones_t.bool()] = 0.0
        expected_values = rewards_v + self.gamma * next_state_values

        loss = self.criteria(state_action_values, expected_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.update_target_steps == 0:
            self.target_model.load_state_dict(self.model.state_dict())
