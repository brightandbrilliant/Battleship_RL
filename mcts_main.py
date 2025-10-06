import numpy as np
import time
import copy  # 需要 deepcopy 来创建环境副本
from battleship_env import CustomBattleshipEnv
from mcts_agent import MCTSAgent
import random

# --- 1. 配置参数 ---
N = 5  # 网格大小 N x N。注意：MCTS 在 N=10 时计算量巨大，推荐从 N=5 开始测试！
N_ARMS = N * N  # 动作空间大小 (25)
TOTAL_EVAL_GAMES = 100  # 总共评估的回合数
MCTS_SIMULATION_TIME = 1.0  # 每次射击 MCTS 搜索的最大时间（秒）

MCTS_PARAMS = {
    'c_param': 1.0,  # UCB1 公式中的探索常数
    'simulation_time': MCTS_SIMULATION_TIME,
}


# --- 2. 辅助函数：获取合法动作掩码 ---

def get_legal_actions_mask(obs_grid):
    """
    根据归一化的观测网格，返回哪些格子（动作）还没有被炸过。
    """
    # 0.0 代表未探索的格子
    return (obs_grid == 0.0).ravel()


# --- 3. MCTS 评估主函数 ---

def evaluate_mcts():
    # 1. 初始化环境（N=5 对应总船体数 2+3+4+5=14）
    # 确保您的 env.py 中的船只配置适应 N=5
    env = CustomBattleshipEnv(N=N)

    # 2. 初始化 MCTS 智能体
    agent = MCTSAgent(N, MCTS_PARAMS, simulation_time=MCTS_SIMULATION_TIME)

    print("-" * 50)
    print(f"模式: MCTS 评估 | 网格尺寸: {N}x{N}")
    print(f"MCTS 搜索时间: {MCTS_SIMULATION_TIME}s/步 | 评估回合数: {TOTAL_EVAL_GAMES}")
    print("-" * 50)

    step_history = []

    for episode in range(TOTAL_EVAL_GAMES):
        # MCTS 不需要训练循环，每次都是一个独立的评估回合
        obs = env.reset() / 2.0  # <-- 观测归一化
        done = False
        total_steps = 0

        start_time = time.time()

        while not done:
            # 1. 获取合法动作掩码
            legal_mask = get_legal_actions_mask(obs)

            # 2. 智能体使用 MCTS 规划选择动作
            # 注意：我们将环境副本传入 act，以便 MCTS 进行模拟
            action = agent.act(copy.deepcopy(env), obs, legal_mask)

            # 3. 环境执行动作
            next_obs, reward, done, info = env.step(action)
            next_obs = next_obs / 2.0  # <-- 归一化下一个状态

            # 4. 更新状态
            obs = next_obs
            total_steps += 1

            # 强制避免过度循环
            if total_steps > N * N * 3:
                print(f"警告：回合 {episode} 步数过多，提前终止。")
                break

        end_time = time.time()
        episode_duration = end_time - start_time

        step_history.append(total_steps)

        # --- 打印进度 ---
        if episode % 10 == 0 or episode == TOTAL_EVAL_GAMES - 1:
            print(f"评估回合 {episode + 1:>3}/{TOTAL_EVAL_GAMES}: "
                  f"步数={total_steps:2d} | "
                  f"总耗时={episode_duration:.2f}s | "
                  f"平均耗时/步={episode_duration / total_steps:.3f}s")

    # 评估结束
    avg_steps = np.mean(step_history)
    print("\n" + "=" * 50)
    print(f"MCTS 评估完成。")
    print(f"平均击沉步数: {avg_steps:.2f} / 目标船体数: {env.total_ship_cells}")
    print("=" * 50)

    return step_history


if __name__ == '__main__':
    # 为了 MCTS 模拟的可重复性，设置随机种子
    np.random.seed(42)
    random.seed(42)  # MCTS 内部的随机 Rollout

    evaluate_mcts()