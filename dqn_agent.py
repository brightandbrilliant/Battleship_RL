import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque


# --- 1. 经验回放缓冲区 (不变) ---

class ReplayBuffer:
    """用于存储和随机采样经验元组 (s, a, r, s', done)"""

    # ... (代码与之前版本完全相同) ...
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """添加经验到缓冲区"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """随机采样一批经验"""
        experiences = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.as_tensor(np.array(states), dtype=torch.float).unsqueeze(1)
        next_states = torch.as_tensor(np.array(next_states), dtype=torch.float).unsqueeze(1)

        actions = torch.as_tensor(np.array(actions), dtype=torch.long)
        rewards = torch.as_tensor(np.array(rewards), dtype=torch.float)
        dones = torch.as_tensor(np.array(dones), dtype=torch.float)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# --- 2. 残差块定义 ---

class ResidualBlock(nn.Module):
    """标准的残差块，包含两个卷积层和跳跃连接"""

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        # 块内第一层：Conv -> BN -> ReLU
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 块内第二层：Conv -> BN
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 跳跃连接（Identity）：如果输入输出通道不一致，使用 1x1 卷积调整维度
        self.identity = nn.Sequential()
        if in_channels != out_channels:
            self.identity = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity_x = self.identity(x)

        # 残差路径
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # 跳跃连接
        out += identity_x
        return F.relu(out)  # 块的最终输出应用 ReLU


# --- 3. 深度 Q-网络 (残差 CNN 结构) ---

class QNetwork(nn.Module):
    """
    使用残差块构建的深层 Q-网络 (共 8 层卷积)
    """

    def __init__(self, N, output_size):
        super(QNetwork, self).__init__()

        # 1. Stem (输入层 - 1 Conv)
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # 2. 残差块 (共 3 个块，每块 2 Conv -> 6 Conv)
        # ResBlock 1: 32 -> 32
        self.res_block1 = ResidualBlock(32, 32)
        # ResBlock 2: 32 -> 64 (增加通道数)
        self.res_block2 = ResidualBlock(32, 64)
        # ResBlock 3: 64 -> 64
        self.res_block3 = ResidualBlock(64, 64)

        # 3. 最终卷积层 (1 Conv)
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # 总卷积层数: 1 (Stem) + 2*3 (Blocks) + 1 (Final) = 8 层

        # 4. 动态计算展平后的维度
        self._to_linear = self._get_conv_output((1, N, N))

        # 5. 全连接层 (头部)
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc_q = nn.Linear(512, output_size)

    def _get_conv_output(self, shape):
        """计算卷积层输出的展平维度"""
        # 模拟前向传播以计算尺寸
        with torch.no_grad():
            o = self.stem(torch.zeros(1, *shape))
            o = self.res_block1(o)
            o = self.res_block2(o)
            o = self.res_block3(o)
            o = self.final_conv(o)
            return int(np.prod(o.size()))

    def forward(self, x):
        # 确保输入是 float 类型
        x = x.float()

        # 卷积层部分
        x = self.stem(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.final_conv(x)

        # 展平特征图
        x = x.view(x.size(0), -1)

        # 全连接层部分
        x = F.relu(self.fc1(x))
        q_values = self.fc_q(x)
        return q_values


# --- 4. DQN 智能体 (与之前版本保持一致) ---

class DQNAgent:
    def __init__(self, N, action_space_size, params, device):
        """
        初始化 DQN 智能体，并设置设备。
        """
        self.N = N
        self.action_space_size = action_space_size
        self.params = params
        self.device = device

        # 超参数 (略)
        self.gamma = params.get('gamma', 0.99)
        self.lr = params.get('lr', 1e-4)
        self.epsilon = params.get('epsilon_start', 1.0)
        self.epsilon_end = params.get('epsilon_end', 0.01)
        self.epsilon_decay = params.get('epsilon_decay', 0.995)
        self.batch_size = params.get('batch_size', 32)
        self.target_update_freq = params.get('target_update_freq', 100)
        self.update_count = 0

        # 初始化 Q-网络 (使用新的 QNetwork)
        self.policy_net = QNetwork(N, action_space_size).to(self.device)
        self.target_net = QNetwork(N, action_space_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = ReplayBuffer(params.get('buffer_capacity', 10000))

    def act(self, obs, legal_actions_mask):
        """根据 epsilon-greedy 策略选择一个动作。"""
        if np.random.rand() < self.epsilon:
            legal_actions = np.where(legal_actions_mask)[0]
            action = np.random.choice(legal_actions)
        else:
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
        """从缓冲区采样，计算 TD 目标，并更新 Policy Net。"""
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        q_current = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(-1)

        with torch.no_grad():
            next_q_max = self.target_net(next_states).max(1)[0]

        q_target = rewards + self.gamma * next_q_max * (1 - dones)
        loss = F.mse_loss(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self._update_target_net()
        self.update_count += 1
        return loss.item()

    def _update_target_net(self):
        if self.update_count > 0 and self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
