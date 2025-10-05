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
        # 注意: state 和 next_state 已经是 NumPy 数组形式
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """随机采样一批经验"""
        experiences = random.sample(self.buffer, batch_size)

        # 将经验元组转换为 NumPy 数组，然后转换为 PyTorch Tensor
        states, actions, rewards, next_states, dones = zip(*experiences)

        # 将 states/next_states 转换为 float，并添加 channel 维度 (B x 1 x N x N)
        states = torch.as_tensor(np.array(states), dtype=torch.float).unsqueeze(1)
        next_states = torch.as_tensor(np.array(next_states), dtype=torch.float).unsqueeze(1)

        # actions/rewards/dones 转换为各自合适的类型
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
        """
        N: 网格边长
        output_size: 动作空间大小 (N*N)
        """
        super(QNetwork, self).__init__()

        # 输入: BATCH x 1 x N x N

        # 1. 卷积层：用于提取空间特征
        # 假设 N=10，经过两个卷积层后，特征图尺寸会缩小
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # 2. 计算展平后的维度 (假设 N=10，经过两个 3x3 卷积，特征图尺寸保持 10x10)
        # 64 channels * 10 * 10 = 6400
        # 动态计算展平维度，以适应不同的 N
        self._to_linear = self._get_conv_output((1, N, N))

        # 3. 全连接层：用于计算最终的 Q 值
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc_q = nn.Linear(512, output_size)

    def _get_conv_output(self, shape):
        """计算卷积层输出的展平维度"""
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
    def __init__(self, N, action_space_size, params):
        """
        初始化 DQN 智能体。
        """
        self.N = N
        self.action_space_size = action_space_size
        self.params = params

        # 超参数
        self.gamma = params.get('gamma', 0.99)
        self.lr = params.get('lr', 1e-4)
        self.epsilon = params.get('epsilon_start', 1.0)
        self.epsilon_end = params.get('epsilon_end', 0.01)
        self.epsilon_decay = params.get('epsilon_decay', 0.995)
        self.batch_size = params.get('batch_size', 32)
        self.target_update_freq = params.get('target_update_freq', 100)
        self.update_count = 0

        # 初始化 在线 Q-网络 和 目标 Q-网络
        self.policy_net = QNetwork(N, action_space_size)
        self.target_net = QNetwork(N, action_space_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 目标网络只用于计算 Q-target，不进行训练

        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # 经验回放缓冲区
        self.memory = ReplayBuffer(params.get('buffer_capacity', 10000))

    def act(self, obs, legal_actions_mask):
        """
        根据 epsilon-greedy 策略选择一个动作。

        参数:
        obs (np.array): 当前状态观测 (N x N)。
        legal_actions_mask (np.array): 长度为 N*N 的布尔数组，True 表示动作合法。
        """
        if np.random.rand() < self.epsilon:
            # 探索: 从合法动作中随机选择
            legal_actions = np.where(legal_actions_mask)[0]
            action = np.random.choice(legal_actions)
        else:
            # 利用: 从 Policy Net 获取 Q 值，并选择 Q 值最高的合法动作
            # 1. 观测转换为 Tensor 并添加 Batch 和 Channel 维度
            obs_tensor = torch.as_tensor(obs, dtype=torch.float).unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                q_values = self.policy_net(obs_tensor).squeeze(0).cpu().numpy()

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

        # 2. 计算 Q(s, a) - 实际执行动作的预测 Q 值 (Policy Net)
        # gather(dim=1, index=actions.unsqueeze(1)) 提取每个样本对应的 action 的 Q 值
        q_current = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(-1)

        # 3. 计算 Q-Target (使用 Target Net)
        # 贝尔曼方程: Q_target = r + gamma * max(Q_target(s', a')) * (1-done)

        # 3a. 获取下一状态的最大 Q 值
        with torch.no_grad():
            # target_net.max(1)[0] 返回每行 (每个样本) 的最大 Q 值
            next_q_max = self.target_net(next_states).max(1)[0]

        # 3b. 计算目标 Q 值
        q_target = rewards + self.gamma * next_q_max * (1 - dones)  # done=1 时, Q-target=r

        # 4. 计算损失 (均方误差 MSE)
        loss = F.mse_loss(q_current, q_target)

        # 5. 反向传播与优化
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪 (可选，用于提高稳定性)
        # nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)

        self.optimizer.step()

        # 6. 更新 Epsilon (探索率衰减)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # 7. 软/硬更新 Target Net
        self._update_target_net()

        self.update_count += 1
        return loss.item()

    def _update_target_net(self):
        """硬更新 Target Net (每 C 步将 Policy Net 权重复制给 Target Net)"""
        if self.update_count % self.target_update_freq == 0 and self.update_count > 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        """保存模型权重"""
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        """加载模型权重"""
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())
