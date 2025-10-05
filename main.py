import numpy as np
import torch
import time
from battleship_env import CustomBattleshipEnv  # 导入环境
from dqn_agent import DQNAgent  # 导入智能体

# --- 1. 配置参数 ---
N = 10  # 网格大小 N x N (10x10)
N_ARMS = N * N  # 动作空间大小 (100)

# DQN 超参数
DQN_PARAMS = {
    'lr': 1e-4,  # 学习率
    'gamma': 0.99,  # 折扣因子
    'epsilon_start': 1.0,  # 初始探索率
    'epsilon_end': 0.05,  # 最终探索率
    'epsilon_decay': 0.9999,  # 探索率衰减系数 (每一步衰减)
    'buffer_capacity': 50000,  # 经验回放缓冲区容量
    'batch_size': 64,  # 每次训练采样的批次大小
    'target_update_freq': 500,  # 目标网络更新频率
    'total_episodes': 10000,  # 总共训练的回合数
}

# 设定设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 2. 辅助函数：获取合法动作掩码 ---

def get_legal_actions_mask(obs_grid):
    """
    根据观测网格，返回哪些格子（动作）还没有被炸过。

    参数:
    obs_grid (N x N np.array): 当前观测，0 代表未探索。

    返回:
    np.array (bool): 长度为 N*N 的布尔数组，True 表示动作合法（未被炸过）。
    """
    # 动作不合法当 obs_grid[i, j] != 0 (即已是 1 或 2)
    # (obs_grid == 0) 返回布尔矩阵，再展平
    return (obs_grid == 0).ravel()


# --- 3. 训练主函数 ---

def train_dqn():
    # 1. 初始化环境和智能体
    env = CustomBattleshipEnv(N=N)
    # 将 DEVICE 传入 DQNAgent
    agent = DQNAgent(N, N_ARMS, DQN_PARAMS, device=DEVICE)

    print("-" * 50)
    print(f"使用的设备: {DEVICE}")
    print(f"网格尺寸: {N}x{N} | 潜水艇总单元数: {env.total_ship_cells}")
    print(f"--- DQN 训练开始 (总回合数={DQN_PARAMS['total_episodes']}) ---")
    print("-" * 50)

    reward_history = []

    for episode in range(DQN_PARAMS['total_episodes']):
        # 重置环境，开始新回合
        obs = env.reset()
        done = False
        total_reward = 0
        total_steps = 0

        while not done:
            # 1. 获取合法动作掩码
            legal_mask = get_legal_actions_mask(obs)

            # 2. 智能体选择动作
            action = agent.act(obs, legal_mask)

            # 3. 环境执行动作
            next_obs, reward, done, info = env.step(action)

            # 4. 存储经验
            agent.store_transition(obs, action, reward, next_obs, done)

            # 5. 更新状态
            obs = next_obs
            total_reward += reward
            total_steps += 1

            # 6. 训练网络 (每步都尝试学习)
            loss = agent.learn()

            # 强制避免过度探索或学不到东西导致死循环
            if total_steps > N * N * 3:
                break

        reward_history.append(total_reward)

        # --- 打印进度和评估 ---
        if episode % 100 == 0 and episode > 0:
            avg_reward = np.mean(reward_history[-100:])
            # 步数是主要的性能指标（我们想让步数尽可能接近潜水艇总数）
            print(f"回合 {episode:>5}: "
                  f"平均奖励={avg_reward:7.2f} | "
                  f"本回合步数={total_steps:3d} | "
                  f"Eps={agent.epsilon:.4f} | "
                  f"Loss={loss:8.4f}" if loss is not None else "Loss=N/A")

    # 训练结束
    model_path = "dqn_battleship_model.pth"
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

    train_dqn()
