import gym
from gym import spaces
import numpy as np


class CustomBattleshipEnv(gym.Env):
    """
    N x N 网格上的潜水艇搜索强化学习环境。
    目标：用尽可能少的炮弹击沉所有潜水艇。
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, N=10, ship_lengths=[5, 4, 3, 3, 2]):
        """
        初始化环境参数。

        参数:
        N (int): 网格尺寸，N x N。
        ship_lengths (list): 潜水艇的长度列表。
        """
        super(CustomBattleshipEnv, self).__init__()

        self.N = N
        self.ship_lengths = ship_lengths
        self.total_ship_cells = sum(ship_lengths)

        # 定义动作空间: N*N 个离散动作，每个动作对应一个格子 (0 到 N*N-1)
        self.action_space = spaces.Discrete(N * N)

        # 定义观测空间: N x N 网格。值域 [0, 2]
        # 0: 未知 (未发射炮弹)
        # 1: 错过 (Miss, 击中水花)
        # 2: 命中 (Hit, 击中船体)
        self.observation_space = spaces.Box(low=0, high=2, shape=(N, N), dtype=np.int8)

        # 内部状态（将在 reset 中初始化）
        self.ship_map = None  # N x N 隐藏地图 (存储船ID或0)
        self.obs_grid = None  # N x N 智能体观测到的网格 (0, 1, 2)
        self.ship_hits = None  # 跟踪每艘潜水艇的被击中次数
        self.hits_left = None  # 剩余未被击中的船体格子数

    def _place_ships(self):
        """
        私有方法：随机且合法地放置所有潜水艇。
        """
        grid = np.zeros((self.N, self.N), dtype=np.int8)
        ship_data = []  # 存储每艘船的 (长度, 命中次数)

        for ship_id, length in enumerate(self.ship_lengths, 1):
            placed = False
            attempts = 0
            while not placed and attempts < 1000:
                attempts += 1
                # 随机选择方向：0=水平, 1=垂直
                direction = np.random.randint(2)

                if direction == 0:  # 水平
                    max_start_row = self.N
                    max_start_col = self.N - length
                else:  # 垂直
                    max_start_row = self.N - length
                    max_start_col = self.N

                start_row = np.random.randint(max_start_row)
                start_col = np.random.randint(max_start_col)

                # 确定船体占用的坐标
                coords = []
                for i in range(length):
                    if direction == 0:
                        coords.append((start_row, start_col + i))
                    else:
                        coords.append((start_row + i, start_col))

                # 检查是否重叠（只检查船体，不检查周围的“水域”隔离，简化处理）
                is_overlap = False
                for r, c in coords:
                    if grid[r, c] != 0:
                        is_overlap = True
                        break

                if not is_overlap:
                    # 放置成功
                    for r, c in coords:
                        grid[r, c] = ship_id
                    ship_data.append(0)  # 初始命中次数为 0
                    placed = True

            if not placed:
                # 如果尝试次数过多仍未放置成功，则重置整个地图并重新开始
                # 这是一个简单的错误处理，大型地图可能需要更优化的放置算法
                return self._place_ships()

        self.ship_map = grid
        self.ship_hits = np.array(ship_data)  # [ship1_hits, ship2_hits, ...]
        self.hits_left = self.total_ship_cells  # 总共需要击中的格子数

        return True

    def reset(self):
        """
        重置环境，开始一个新的回合。
        """
        # 1. 随机放置潜水艇，并初始化内部状态
        self._place_ships()

        # 2. 初始化观测网格 (全 0: 未知)
        self.obs_grid = np.zeros((self.N, self.N), dtype=np.int8)

        return self.obs_grid

    def step(self, action):
        """
        执行一步动作（开炮）。

        参数:
        action (int): 从 0 到 N*N-1 的离散动作索引。

        返回:
        (obs, reward, done, info)
        """
        row, col = np.unravel_index(action, (self.N, self.N))

        # 默认奖励为弹药消耗惩罚
        reward = -1
        done = False
        info = {}

        # --- 1. 处理重复动作 ---
        if self.obs_grid[row, col] != 0:
            # 这一格已经被炸过 (obs_grid 值不是 0)
            # 强化学习的目标是学习避免这种浪费，给予一个巨大的负面奖励
            reward -= 1000
            return self.obs_grid, reward, done, info

        # --- 2. 检查是否命中潜水艇 ---
        ship_id = self.ship_map[row, col]

        if ship_id > 0:
            # 命中！(Hit)
            self.obs_grid[row, col] = 2  # 观测更新为 命中
            reward += 10  # 给予命中奖励

            # 跟踪船体受损情况
            ship_index = ship_id - 1
            self.ship_hits[ship_index] += 1
            self.hits_left -= 1

            # 检查是否击沉
            ship_length = self.ship_lengths[ship_index]
            if self.ship_hits[ship_index] == ship_length:
                # 击沉！
                reward += 100  # 给予击沉高奖励
                info['sunk'] = ship_id

        else:
            # 未命中 (Miss)
            self.obs_grid[row, col] = 1  # 观测更新为 错过
            reward += 0

        # --- 3. 检查终止条件 ---
        if self.hits_left == 0:
            done = True
            # 给予游戏结束的大奖励
            reward += 500

        return self.obs_grid, reward, done, info

    # 我们暂时不实现 render 方法，但如果需要可视化，可以后续添加
    # def render(self, mode='human'):
    #     pass

    # def close(self):
    #     pass

# --- 简单测试环境 ---
# env = CustomBattleshipEnv()
# obs = env.reset()
# print("初始观测网格:\n", obs)
#
# # 尝试发射几炮 (例如，炸开 (0, 0) 和 (9, 9))
# obs, reward, done, info = env.step(action=0 * env.N + 0) # 0, 0
# print(f"\n动作: (0, 0), 奖励: {reward}, Done: {done}")
# obs, reward, done, info = env.step(action=9 * env.N + 9) # 9, 9
# print(f"动作: (9, 9), 奖励: {reward}, Done: {done}")
# print("当前观测网格:\n", obs)
