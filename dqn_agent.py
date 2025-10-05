import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque


# --- 1. 经验回放缓冲区 ---

class ReplayBuffer:
    """用于存储和随机采样经验元组 (s, a, r, s', done)"""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """添加经验到缓冲区"""
        # 注意: state 和 next_state 存储的是 NumPy 数组
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """随机采样一批经验"""
        experiences = random.sample(self.buffer, batch_size)

        # 将经验元组转换为 NumPy 数组
        states, actions, rewards, next_states, dones = zip(*experiences)

        # 将 NumPy 数组转换为 PyTorch Tensor (注意：不在这里移动到设备)
        # states/next_states: B x 1 x N x N (float)
        states = torch.as_tensor(np.array(states), dtype=torch.float).unsqueeze(1)
        next_states = torch.as_tensor(np.array(next_states), dtype=torch.float).unsqueeze(1)

        # actions/rewards/dones: B (long/float)
        actions = torch.as_tensor(np.array(actions), dtype=torch.long)
        rewards = torch.as_tensor(np.array(rewards), dtype=torch.float)
        dones = torch.as_tensor(np.array(dones), dtype=torch.float)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# --- 2. 深度 Q-网络 (CNN 结构) ---

class QNetwork(nn.Module):
    """
    用于 Battleship 游戏的 Q-网络 (使用 CNN 处理网格输入)
    """

    def __init__(self, N, output_size):
        super(QNetwork, self).__init__()

        # 1. 卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # 2. 动态计算展平后的维度
        self._to_linear = self._get_conv_output((1, N, N))

        # 3. 全连接层
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc_q = nn.Linear(512, output_size)

    def _get_conv_output(self, shape):
        """计算卷积层输出的展平维度"""
        # 使用一个 dummy tensor 来跟踪尺寸变化
        o = F.relu(self.conv1(torch.zeros(1, *shape)))
        o = F.relu(self.conv2(o))
        return int(np.prod(o.size()))

    def forward(self, x):
        # 确保输入是 float 类型
        x = x.float()

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # 展平特征图
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        q_values = self.fc_q(x)
        return q_values


# --- 3. DQN 智能体 ---

class DQNAgent:
    def __init__(self, N, action_space_size, params, device):
        """
        初始化 DQN 智能体，并设置设备。
        """
        self.N = N
        self.action_space_size = action_space_size
        self.params = params
        self.device = device  # <--- 新增: 存储设备信息

        # 超参数
        self.gamma = params.get('gamma', 0.99)
        self.lr = params.get('lr', 1e-4)
        self.epsilon = params.get('epsilon_start', 1.0)
        self.epsilon_end = params.get('epsilon_end', 0.01)
        self.epsilon_decay = params.get('epsilon_decay', 0.995)
        self.batch_size = params.get('batch_size', 32)
        self.target_update_freq = params.get('target_update_freq', 100)
        self.update_count = 0

        # 初始化 在线 Q-网络 和 目标 Q-网络，并移动到设备
        self.policy_net = QNetwork(N, action_space_size).to(self.device)  # <--- 移动到设备
        self.target_net = QNetwork(N, action_space_size).to(self.device)  # <--- 移动到设备
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 目标网络不训练

        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # 经验回放缓冲区
        self.memory = ReplayBuffer(params.get('buffer_capacity', 10000))

    def act(self, obs, legal_actions_mask):
        """
        根据 epsilon-greedy 策略选择一个动作。
        """
        if np.random.rand() < self.epsilon:
            # 探索: 从合法动作中随机选择
            legal_actions = np.where(legal_actions_mask)[0]
            action = np.random.choice(legal_actions)
        else:
            # 利用: 从 Policy Net 获取 Q 值，并选择 Q 值最高的合法动作
            # 1. 观测转换为 Tensor 并移动到设备
            obs_tensor = (torch.as_tensor(obs, dtype=torch.float)
                          .unsqueeze(0).unsqueeze(0)
                          .to(self.device))  # <--- 移动到设备

            with torch.no_grad():
                q_values = self.policy_net(obs_tensor).squeeze(0).cpu().numpy()  # <--- 移回CPU进行Numpy操作

            # 2. 应用动作屏蔽：将不合法动作的 Q 值设为极小值
            q_values[~legal_actions_mask] = -float('inf')

            # 3. 选择 Q 值最大的动作
            action = np.argmax(q_values)

        return action

    def store_transition(self, state, action, reward, next_state, done):
        """将经验存储到回放缓冲区。"""
        self.memory.push(state, action, reward, next_state, done)

    def learn(self):
        """
        从缓冲区采样，计算 TD 目标，并更新 Policy Net。
        """
        if len(self.memory) < self.batch_size:
            return  # 缓冲区未满，不进行训练

        # 1. 从缓冲区采样
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # 将所有采样数据移动到设备上！
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # 2. 计算 Q(s, a) - 实际执行动作的预测 Q 值 (Policy Net)
        q_current = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(-1)

        # 3. 计算 Q-Target (使用 Target Net)
        with torch.no_grad():
            next_q_max = self.target_net(next_states).max(1)[0]

        q_target = rewards + self.gamma * next_q_max * (1 - dones)

        # 4. 计算损失 (均方误差 MSE)
        loss = F.mse_loss(q_current, q_target)

        # 5. 反向传播与优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 6. 更新 Epsilon (探索率衰减)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # 7. 硬更新 Target Net
        self._update_target_net()

        self.update_count += 1
        return loss.item()

    def _update_target_net(self):
        """硬更新 Target Net (每 C 步将 Policy Net 权重复制给 Target Net)"""
        if self.update_count > 0 and self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        """保存模型权重"""
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        """加载模型权重"""
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
