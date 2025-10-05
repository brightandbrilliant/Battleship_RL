import numpy as np
import torch
import time
from battleship_env import CustomBattleshipEnv  # 导入环境
from dqn_agent import DQNAgent  # 导入智能体

# --- 1. 配置参数 ---
N = 10  # 网格大小 N x N (10x10 是经典战舰尺寸)
N_ARMS = N * N  # 动作空间大小

# DQN 超参数
DQN_PARAMS = {
    'lr': 1e-4,  # 学习率
    'gamma': 0.99,  # 折扣因子
    'epsilon_start': 1.0,  # 初始探索率
    'epsilon_end': 0.05,  # 最终探索率
    'epsilon_decay': 0.9995,  # 探索率衰减系数 (每一步衰减)
    'buffer_capacity': 50000,  # 经验回放缓冲区容量
    'batch_size': 64,  # 每次训练采样的批次大小
    'target_update_freq': 500,  # 目标网络更新频率 (每 500 步更新一次)
    'total_episodes': 5000,  # 总共训练的回合数
}


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
    # np.ravel() 将 N x N 展平为 N*N
    # (obs_grid == 0) 返回布尔矩阵，再展平
    return (obs_grid == 0).ravel()


# --- 3. 训练主函数 ---

def train_dqn():
    # 1. 初始化环境和智能体
    env = CustomBattleshipEnv(N=N)
    agent = DQNAgent(N, N_ARMS, DQN_PARAMS)

    print(f"--- DQN 训练开始 (N={N}x{N}, 总回合数={DQN_PARAMS['total_episodes']}) ---")
    print(f"最优沉船弹药消耗理论上限: {env.total_ship_cells}")

    reward_history = []

    for episode in range(DQN_PARAMS['total_episodes']):
        # 重置环境，开始新回合
        obs = env.reset()
        done = False
        total_reward = 0
        total_steps = 0

        start_time = time.time()

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

            # 强制避免无限循环（防止策略学歪了）
            if total_steps > N * N * 2:  # 比如最多尝试 2 * N^2 步
                break

        end_time = time.time()
        reward_history.append(total_reward)

        # --- 打印进度和评估 ---
        if episode % 100 == 0:
            avg_reward = np.mean(reward_history[-100:])
            # 弹药消耗 ≈ -total_reward (因为主要奖励是 -1)
            # 实际消耗： 总步数 - (命中奖励 + 击沉奖励)
            print(f"回合 {episode:>4}: "
                  f"平均奖励={avg_reward:7.2f} | "
                  f"本回合步数={total_steps:3d} | "
                  f"Eps={agent.epsilon:.4f} | "
                  f"Loss={loss:8.4f}" if loss is not None else "Loss=N/A")

    # 训练结束
    agent.save("dqn_battleship_model.pth")
    print("\n训练完成。模型已保存到 dqn_battleship_model.pth。")

    # 返回历史记录，可选用于绘图
    return reward_history


if __name__ == '__main__':
    # 设置 PyTorch 随机种子以保证可复现性
    torch.manual_seed(42)
    np.random.seed(42)

    train_dqn()
