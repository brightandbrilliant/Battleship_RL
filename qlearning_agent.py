import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque


# --- 1. 经验回放缓冲区 (保持不变) ---

class ReplayBuffer:
    """用于存储和随机采样经验元组 (s, a, r, s', done)"""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*experiences)

        # 将 NumPy 数组转换为 PyTorch Tensor
        states = torch.as_tensor(np.array(states), dtype=torch.float).unsqueeze(1)
        next_states = torch.as_tensor(np.array(next_states), dtype=torch.float).unsqueeze(1)

        actions = torch.as_tensor(np.array(actions), dtype=torch.long)
        rewards = torch.as_tensor(np.array(rewards), dtype=torch.float)
        dones = torch.as_tensor(np.array(dones), dtype=torch.float)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# --- 2. 残差块和 Q-网络 (保持不变) ---

class ResidualBlock(nn.Module):
    # ... (代码与之前的 ResidualBlock 完全相同) ...
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.identity = nn.Sequential()
        if in_channels != out_channels:
            self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity_x = self.identity(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += identity_x
        return F.relu(out)


class QNetwork(nn.Module):
    # ... (代码与之前的 QNetwork 完全相同) ...
    def __init__(self, N, output_size):
        super(QNetwork, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.res_block1 = ResidualBlock(32, 32)
        self.res_block2 = ResidualBlock(32, 64)
        self.res_block3 = ResidualBlock(64, 64)

        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self._to_linear = self._get_conv_output((1, N, N))

        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc_q = nn.Linear(512, output_size)

    def _get_conv_output(self, shape):
        with torch.no_grad():
            o = self.stem(torch.zeros(1, *shape))
            o = self.res_block1(o)
            o = self.res_block2(o)
            o = self.res_block3(o)
            o = self.final_conv(o)
            return int(np.prod(o.size()))

    def forward(self, x):
        x = x.float()

        x = self.stem(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.final_conv(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        q_values = self.fc_q(x)
        return q_values


# --- 3. 深度 Q-Learning 智能体 (移除 Target Net) ---

class DeepQLearningAgent:  # <-- 更改了类名
    def __init__(self, N, action_space_size, params, device):
        """
        初始化 DQL 智能体，**不使用目标网络**。
        """
        self.N = N
        self.action_space_size = action_space_size
        self.params = params
        self.device = device

        # 超参数
        self.gamma = params.get('gamma', 0.99)
        self.lr = params.get('lr', 1e-4)
        self.epsilon = params.get('epsilon_start', 1.0)
        self.epsilon_end = params.get('epsilon_end', 0.01)
        self.epsilon_decay = params.get('epsilon_decay', 0.995)
        self.batch_size = params.get('batch_size', 32)
        # self.target_update_freq = 0 # <-- 目标网络已移除
        self.update_count = 0

        # 初始化 在线 Q-网络 (Policy Net)
        self.policy_net = QNetwork(N, action_space_size).to(self.device)  # <-- 只有一个网络

        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # 经验回放缓冲区
        self.memory = ReplayBuffer(params.get('buffer_capacity', 10000))

    def act(self, obs, legal_actions_mask):
        """根据 epsilon-greedy 策略选择一个动作。"""
        if np.random.rand() < self.epsilon:
            # 探索
            legal_actions = np.where(legal_actions_mask)[0]
            action = np.random.choice(legal_actions)
        else:
            # 利用
            obs_tensor = (torch.as_tensor(obs, dtype=torch.float)
                          .unsqueeze(0).unsqueeze(0)
                          .to(self.device))

            with torch.no_grad():
                q_values = self.policy_net(obs_tensor).squeeze(0).cpu().numpy()

            q_values[~legal_actions_mask] = -float('inf')
            action = np.argmax(q_values)

        return action

    def store_transition(self, state, action, reward, next_state, done):
        """将经验存储到回放缓冲区。"""
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        """
        从缓冲区采样，计算 TD 目标，并更新 Policy Net (Q-Learning 更新)。
        """
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # 2. 计算 Q(s, a) - 实际执行动作的预测 Q 值 (Policy Net)
        q_current = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(-1)

        # 3. 计算 Q-Target (使用 Q-Learning 公式，不使用 Target Net)
        # Q_target = r + gamma * max(Q_policy(s', a')) * (1-done)
        with torch.no_grad():  # 保持稳定性，目标侧不计算梯度
            # 使用 policy_net 来计算下一状态的 Q 值
            next_q_max = self.policy_net(next_states).max(1)[0]  # <-- 关键改变：使用 policy_net

        q_target = rewards + self.gamma * next_q_max * (1 - dones)

        # 4. 计算损失 (均方误差 MSE)
        loss = F.mse_loss(q_current, q_target)

        # 5. 反向传播与优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 6. 更新 Epsilon (探索率衰减)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        self.update_count += 1
        return loss.item()

    # 目标网络已移除，故无需 _update_target_net
    # def _update_target_net(self):
    #     pass

    def save(self, path):
        """保存模型权重"""
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        """加载模型权重"""
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        # 目标网络已移除，无需加载目标网络
        