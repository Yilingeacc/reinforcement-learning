import torch
import numpy as np


class BufferBase:
    def __init__(self, capacity, state_dim, continuous=False):
        self.capacity = capacity
        self.mem_cntr = 0
        self.is_full = False

        self.states = torch.empty((capacity, state_dim), dtype=torch.float32).cuda()
        self.actions = torch.empty((capacity, 1), dtype=torch.float32 if continuous else torch.int32).cuda()
        self.rewards = torch.empty((capacity, 1), dtype=torch.float32).cuda()
        self.masks = torch.empty((capacity, 1), dtype=torch.float32).cuda()

    def store(self, state, action, reward, mask):
        pass

    def sample(self, batch_size):
        pass


class ReplayBuffer(BufferBase):
    def __init__(self, capacity, state_dim, continuous=False):
        super().__init__(capacity, state_dim, continuous)

    def store(self, state, action, reward, mask):
        self.states[self.mem_cntr] = state
        self.actions[self.mem_cntr] = action
        self.rewards[self.mem_cntr] = reward
        self.masks[self.mem_cntr] = mask

        self.mem_cntr += 1
        if self.mem_cntr >= self.capacity:
            self.mem_cntr = 0
            self.is_full = True

    def sample(self, batch_size):
        max_memory = self.capacity if self.is_full else self.mem_cntr
        batch = torch.randint(max_memory - 1, size=(batch_size,)).cuda()
        batch = torch.where(batch == self.mem_cntr - 1, batch - 1, batch)
        return self.states[batch], self.actions[batch], self.rewards[batch], self.masks[batch], self.states[batch + 1]


class SumTree:
    """
    SumTree data structure:
    every node is the sum of its children, with the priorities as the leaf nodes
    """

    # Full Binary Tree:
    #                                 42
    #                          /              \
    #                  29                            13
    #               /      \                      /      \
    #          13             16              3              10
    #        /    \         /    \         /     \         /     \
    #       3     10      12      4       1       2       8       2
    #     (0~3) (3~13) (13~25) (25~29) (29~30) (30~32) (32~40) (40~42)
    #
    # 其每个叶子结点代表一个transition的优先级，下面括号所示范围代表当随机数在此范围内，将选取对应transition
    #
    # 取样步骤：将优先级总和划分为batch_size个区间。如上图所示，当batch_size=6时，将均分为6个区间：
    # [0~7] [7~14] [14~21] [21~28] [28~35] [35~42]
    # 然后在各区间上随机取样，并选取对应的transition。
    #
    # 选取步骤：从root开始与左节点对比，如果随机数小于左节点就选取左子树，否则选取右子树。
    # 若选取右节点，则将随机数减去左节点。重复此步骤，直到到达叶子结点。
    #
    # e.g.如果在[21~28]上随机数为24：
    #   24与左节点29对比，小于29，选取左子树29
    #   24与29的左节点13对比，大于13，选取右子树16，新随机数=24-13=11
    #   11与16的左节点12对比，小于12，选取左子树12
    #   到达叶子结点，选取优先级为12的transition

    def __init__(self, capacity):
        self.capacity = capacity
        # Full Binary Tree Structure:
        # [-------Parent nodes-------][-------leaves to recode priority-------]
        #      size: capacity - 1                  size: capacity
        self.tree = np.zeros(2 * capacity - 1)

    def add(self, data_index, p):
        index = data_index + self.capacity - 1
        self.update(index, p)

    def update(self, index, p):
        change = p - self.tree[index]
        self.tree[index] = p
        while index:
            index = (index - 1) // 2
            self.tree[index] += change

    def get(self, random):
        index = 0
        while True:
            left_child_idx = 2 * index + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):
                break
            else:
                if random <= self.tree[left_child_idx]:
                    index = left_child_idx
                else:
                    random -= self.tree[left_child_idx]
                    index = right_child_idx
        return index - self.capacity + 1

    @property
    def total_p(self):
        return self.tree[0]


class PERBuffer(BufferBase):
    epsilon = 0.01  # guaranteeing a non-zero probability
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    # beta = 0.4  # importance-sampling
    # beta_increment = 1e-3  # linearly anneal beta from its initial value to 1
    max_prio = 1.

    def __init__(self, capacity, state_dim, continuous=False):
        super().__init__(capacity, state_dim, continuous)
        self.tree = SumTree(capacity)
        self.batch = None

    def store(self, state, action, reward, mask):
        """
        new transitions arrive without a known TD-error
        so we put them at maximal priority in order to guarantee that all experience is seen at least once
        """
        self.states[self.mem_cntr] = state
        self.actions[self.mem_cntr] = action
        self.rewards[self.mem_cntr] = reward
        self.masks[self.mem_cntr] = mask

        p = np.power(self.max_prio + self.epsilon, self.alpha)
        self.tree.add(self.mem_cntr, p)

        self.mem_cntr += 1
        if self.mem_cntr >= self.capacity:
            self.mem_cntr = 0
            self.is_full = True

    def sample(self, batch_size):
        self.batch = torch.empty(batch_size, dtype=torch.long).cuda() if self.batch is None else self.batch
        segment = self.tree.total_p / batch_size

        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            random = np.random.uniform(a, b)
            index = self.tree.get(random)
            self.batch[i] = index
        self.batch = torch.where((self.batch == self.capacity - 1) | (self.batch == self.mem_cntr - 1),
                                 self.batch - 1, self.batch)
        return self.states[self.batch], self.actions[self.batch], self.rewards[self.batch], self.masks[self.batch], \
               self.states[self.batch + 1]

    def batch_update(self, abs_errors):
        clipped_errors = np.minimum(abs_errors + self.epsilon, self.max_prio)
        ps = np.power(clipped_errors, self.alpha)
        tree_idxs = self.batch.cpu().numpy() + self.capacity - 1
        for idx, p in zip(tree_idxs, ps):
            self.tree.update(idx, p)
