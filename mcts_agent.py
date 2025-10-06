import numpy as np
import random
import copy
import math
import time

# --- 1. MCTS 节点定义 ---

class Node:
    def __init__(self, state, parent=None, parent_action=None):
        self.state = state.copy()  # 观测网格的副本
        self.parent = parent
        self.parent_action = parent_action
        self.children = {}  # action -> Node
        self.visits = 0  # 访问次数 N(s)
        self.value = 0.0  # 累计价值 W(s)
        self.untried_actions = None  # 稍后初始化为所有合法动作

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def is_terminal(self, env):
        # 检查当前状态是否为终止状态 (即所有船都被击沉)
        # 注意: 需要一个方法来检查 state 是否为终止状态。
        # 在 Battleship 中，可以检查环境的 hits_left 状态，但由于 Node 不直接存储 hits_left，
        # 我们需要模拟一步来判断，或者使用一个辅助函数。
        # 简单起见，我们假设 env 提供了一个方法来检查一个 state 是否终止

        # 临时的、简化的检查：如果格子只剩下 1 或 2，理论上就结束了
        # 但准确性依赖于 env 逻辑，这里先跳过，假设 MCTS 依赖模拟来发现终止。
        return False  # 真正的终止检查发生在模拟时

    def best_child(self, c_param=1.0):
        """
        使用 UCB1 公式选择最佳子节点
        UCB1 = Q(s, a) / N(s, a) + C * sqrt(ln(N(s)) / N(s, a))
        """
        best_score = -float('inf')
        best_child = None

        for action, child in self.children.items():
            # Q(s, a) / N(s, a) 是平均价值，即 Exploitation (利用)
            exploitation_term = child.value / child.visits

            # sqrt(ln(N(s)) / N(s, a)) 是 Exploration (探索)
            exploration_term = c_param * math.sqrt(math.log(self.visits) / child.visits)

            score = exploitation_term + exploration_term

            if score > best_score:
                best_score = score
                best_child = child
        return best_child


# --- 2. MCTS 智能体定义 ---

class MCTSAgent:
    def __init__(self, N, params, simulation_time=1.0):
        self.N = N
        self.action_space_size = N * N
        self.c_param = params.get('c_param', 1.0)  # UCB1 参数
        self.simulation_time = simulation_time  # 每次决策的最大思考时间 (秒)
        # MCTS Agent 不需要 epsilon 衰减，因为它使用 UCB1 进行平衡

    def act(self, env, obs, legal_actions_mask):
        """
        主 MCTS 搜索循环，返回最佳动作。
        """
        # 1. 初始化根节点
        root = Node(obs)
        root.untried_actions = np.where(legal_actions_mask)[0].tolist()

        start_time = time.time()
        num_simulations = 0

        # 2. MCTS 搜索循环 (基于时间或迭代次数)
        while time.time() - start_time < self.simulation_time:
            node = root

            # 2a. 选择 (Selection)
            # 从根节点向下，直到找到一个非完全扩展的节点
            while node.is_fully_expanded() and node.children:
                # 在 MCTS 中，子节点的选择是基于 UCB1 的
                node = node.best_child(self.c_param)

            # 2b. 扩展 (Expansion)
            # 如果不是终止节点，则扩展一个未尝试过的子节点
            if node.untried_actions:
                # 随机选择一个未尝试的动作进行扩展
                action_to_try = random.choice(node.untried_actions)
                node.untried_actions.remove(action_to_try)

                # 在真实环境的副本上执行动作，以获得 next_state
                # 注意：MCTS 必须在环境的副本上进行操作，以防止污染真实环境！
                simulation_env = copy.deepcopy(env)
                next_obs, reward, done, _ = simulation_env.step(action_to_try)

                # 创建新的子节点
                node = Node(next_obs, parent=node, parent_action=action_to_try)
                node.untried_actions = np.where((next_obs == 0.0).ravel())[0].tolist()
                root.children[action_to_try] = node  # 将新节点添加到父节点的子节点字典中

            # 2c. 模拟 (Simulation) - 随机 Rollout
            # 从当前节点开始，使用随机策略玩到游戏结束
            rollout_env = copy.deepcopy(simulation_env)
            # 注意: 如果上一步已经结束 (done=True)，则直接使用 reward，不进行模拟
            reward_g = reward if done else self._rollout(rollout_env, node.state)

            # 2d. 反向传播 (Backpropagation)
            self._backpropagate(node, reward_g)
            num_simulations += 1

        # 3. 确定最终动作
        # 搜索结束后，选择访问次数最多的子节点所对应的动作
        best_action = max(root.children.items(), key=lambda item: item[1].visits)[0]

        # print(f"MCTS 完成 {num_simulations} 次模拟。")
        return best_action

    def _rollout(self, env, start_obs):
        """
        使用一个快速（随机）策略玩到游戏结束，并返回累计奖励 (G)。
        """
        current_obs = start_obs
        done = False
        total_reward = 0.0
        # MCTS 模拟中通常不需要折扣因子（gamma=1.0）

        while not done:
            # 快速随机策略：在所有未射击的格子中随机选择一个
            legal_mask = (current_obs == 0.0).ravel()
            legal_actions = np.where(legal_mask)[0]

            if not legal_actions.size:  # 没有合法动作，但还没结束，可能是逻辑错误或步数过多
                break

            action = random.choice(legal_actions)

            current_obs, reward, done, _ = env.step(action)
            current_obs = current_obs / 2.0  # 保持归一化
            total_reward += reward

            # 强制避免无限循环
            if total_reward < -500:  # 如果惩罚过大，提前终止
                break

        return total_reward

    def _backpropagate(self, node, reward):
        """
        将回报反向传播至根节点，更新 N(s) 和 W(s)。
        """
        while node is not None:
            node.visits += 1
            node.value += reward  # 注意: MCTS 通常是累加价值，不是平均

            # 对于父节点，我们需要对回报进行折扣，但在战舰游戏中我们通常不使用折扣 (gamma=1.0)
            # reward = node.gamma * reward # 如果 gamma < 1.0

            node = node.parent
