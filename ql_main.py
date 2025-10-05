import numpy as np
import torch
import time
# 导入环境
from battleship_env import CustomBattleshipEnv
# 导入新的 Q-Learning 智能体
from qlearning_agent import DeepQLearningAgent

# --- 1. 配置参数 ---
N = 10  # 网格大小 N x N (10x10)
N_ARMS = N * N  # 动作空间大小 (100)

# DQL 超参数 (已移除 Target Net 相关的参数)
DQL_PARAMS = {
    'lr': 1e-4,  # 学习率
    'gamma': 0.99,  # 折扣因子
    'epsilon_start': 1.0,  # 初始探索率
    'epsilon_end': 0.05,  # 最终探索率
    'epsilon_decay': 0.9999,  # 探索率衰减系数
    'buffer_capacity': 50000,  # 经验回放缓冲区容量
    'batch_size': 64,  # 每次训练采样的批次大小
    'total_episodes': 10000,  # 总共训练的回合数
}

# 设定设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 2. 辅助函数：获取合法动作掩码 ---

def get_legal_actions_mask(obs_grid):
    """
    根据观测网格，返回哪些格子（动作）还没有被炸过。
    """
    # obs_grid 中的值是归一化后的 (0, 0.5, 1.0)
    # 原始值 0 (未探索) 对应归一化后的 0.0
    return (obs_grid == 0.0).ravel()


# --- 3. 训练主函数 ---

def train_dql():
    # 1. 初始化环境和智能体
    env = CustomBattleshipEnv(N=N)
    # 实例化 Deep Q-Learning Agent
    agent = DeepQLearningAgent(N, N_ARMS, DQL_PARAMS, device=DEVICE)

    print("-" * 50)
    print(f"模式: Deep Q-Learning | 设备: {DEVICE}")
    print(f"网格尺寸: {N}x{N} | 潜水艇总单元数: {env.total_ship_cells}")
    print(f"--- DQL 训练开始 (总回合数={DQL_PARAMS['total_episodes']}) ---")
    print("-" * 50)

    reward_history = []

    for episode in range(DQL_PARAMS['total_episodes']):
        # 重置环境，开始新回合，并进行归一化
        obs = env.reset() / 2.0  # <-- 观测归一化
        done = False
        total_reward = 0
        total_steps = 0

        while not done:
            # 1. 获取合法动作掩码 (基于归一化后的观测)
            legal_mask = get_legal_actions_mask(obs)

            # 2. 智能体选择动作
            action = agent.act(obs, legal_mask)

            # 3. 环境执行动作
            next_obs, reward, done, info = env.step(action)
            # 归一化下一个状态
            next_obs = next_obs / 2.0  # <-- 归一化下一个状态

            # 4. 存储经验
            agent.store_transition(obs, action, reward, next_obs, done)

            # 5. 更新状态
            obs = next_obs
            total_reward += reward
            total_steps += 1

            # 6. 训练网络 (每步都尝试学习)
            loss = agent.learn()

            # 强制避免过度循环
            if total_steps > N * N * 3:
                break

        reward_history.append(total_reward)

        # --- 打印进度和评估 ---
        if episode % 100 == 0 and episode > 0:
            avg_reward = np.mean(reward_history[-100:])
            # 评估指标: 步数越接近 env.total_ship_cells (14)，策略越好
            avg_steps = np.mean([s for s in [total_steps] if s < N * N * 3])

            print(f"回合 {episode:>5}: "
                  f"平均奖励={avg_reward:7.2f} | "
                  f"本回合步数={total_steps:3d} | "
                  f"Eps={agent.epsilon:.4f} | "
                  f"Loss={loss:8.4f}" if loss is not None else "Loss=N/A")

    # 训练结束
    model_path = "dql_battleship_model.pth"
    agent.save(model_path)
    print("\n" + "-" * 50)
    print(f"训练完成。模型已保存到 {model_path}。")
    print(f"最终平均回合奖励: {np.mean(reward_history[-100:]):.2f}")
    print("-" * 50)

    return reward_history


if __name__ == '__main__':
    # 设置随机种子以保证实验可复现性
    torch.manual_seed(42)
    np.random.seed(42)

    train_dql()
