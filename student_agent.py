# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math
import gdown
from collections import defaultdict

gdown.download('https://drive.google.com/uc?id=1Lc-ZuvS-lWQTTWsteMfDYWCv3rPDTVbx', 'weights.pkl', quiet=False)

class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4  # 4x4 2048 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True  # Record if the last move was valid

        self.reset()

    def reset(self):
        """Reset the environment"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        """Compress the row: move non-zero values to the left"""
        new_row = row[row != 0]  # Remove zeros
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')  # Pad with zeros on the right
        return new_row

    def merge(self, row):
        """Merge adjacent equal numbers in the row"""
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        """Move the board left"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        """Move the board right"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            # Reverse the row, compress, merge, compress, then reverse back
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        """Move the board up"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        """Move the board down"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            # Reverse the column, compress, merge, compress, then reverse back
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        """Check if there are no legal moves left"""
        # If there is any empty cell, the game is not over
        if np.any(self.board == 0):
            return False

        # Check horizontally
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False

        # Check vertically
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def step(self, action):
        """Execute one action"""
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved  # Record if the move was valid

        if moved:
            self.add_random_tile()

        done = self.is_game_over()

        return self.board, self.score, done, {}

    def render(self, mode="human", action=None):
        """
        Render the current board using Matplotlib.
        This function does not check if the action is valid and only displays the current board state.
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")  # Default dark color
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def simulate_row_move(self, row):
        """Simulate a left move for a single row"""
        # Compress: move non-zero numbers to the left
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        # Merge: merge adjacent equal numbers (do not update score)
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        # Compress again
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        """Check if the specified move is legal (i.e., changes the board)"""
        # Create a copy of the current board state
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                # Reverse the column, simulate, then reverse back
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")

        # If the simulated board is different from the current board, the move is legal
        return not np.array_equal(self.board, temp_board)

patterns = [
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
    [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (3, 1)],
    [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (2, 2)],
    [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1), (2, 2)],
    [(0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (3, 2)],
    [(0, 1), (0, 2), (1, 1), (2, 1), (3, 1), (2, 0)],
    [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (1, 0)]
]

class NTupleApproximator:
    def __init__(self, board_size, patterns):
        self.base = 16
        self.bases = [self.base ** i for i in range(16)]
        self.board_size = board_size
        self.patterns = patterns
        self.weights = [defaultdict(float) for _ in patterns]
        self.symmetry_patterns = []
        for pattern in self.patterns:
            syms = self.generate_symmetries(pattern)
            self.symmetry_patterns.append(syms)

    def generate_symmetries(self, pattern):
        symmetries = []
        n = self.board_size - 1
        # Original pattern
        symmetries.append(pattern)
        # Rotate 90° clockwise: (x, y) -> (y, n-x)
        rot90 = [(y, n-x) for x, y in pattern]
        symmetries.append(rot90)
        # Rotate 180°: (x, y) -> (n-x, n-y)
        rot180 = [(n-x, n-y) for x, y in pattern]
        symmetries.append(rot180)
        # Rotate 270° clockwise: (x, y) -> (n-y, x)
        rot270 = [(n-y, x) for x, y in pattern]
        symmetries.append(rot270)
        # Horizontal reflection: (x, y) -> (x, n-y)
        h_reflect = [(x, n-y) for x, y in pattern]
        symmetries.append(h_reflect)
        # Vertical reflection: (x, y) -> (n-x, y)
        v_reflect = [(n-x, y) for x, y in pattern]
        symmetries.append(v_reflect)
        # Diagonal reflection (top-left to bottom-right): (x, y) -> (y, x)
        diag1_reflect = [(y, x) for x, y in pattern]
        symmetries.append(diag1_reflect)
        # Diagonal reflection (top-right to bottom-left): (x, y) -> (n-y, n-x)
        diag2_reflect = [(n-y, n-x) for x, y in pattern]
        symmetries.append(diag2_reflect)
        # Remove duplicates by converting to tuples of tuples and using a set
        unique_symmetries = []
        unique_set = set()
        for sym in symmetries:
            # Sort each pattern to ensure consistent ordering
            sorted_sym = sorted(sym)
            tuple_sym = tuple(sorted_sym)
            if tuple_sym not in unique_set:
                unique_set.add(tuple_sym)
                unique_symmetries.append(sorted_sym)
        return unique_symmetries

    def tile_to_index(self, tile):
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, coords):
        return tuple([self.tile_to_index(board[x][y]) for x, y in coords])

    def compute_index(self, values):
        index = 0
        for i, val in enumerate(reversed(values)):
            index += val * self.bases[i]
        return index

    def value(self, board):
        total = []
        for i in range(len(self.patterns)):
            # LUT for this pattern
            LUT = self.weights[i]
            for sym in self.symmetry_patterns[i]:
                tup = self.get_feature(board, sym)
                idx = self.compute_index(tup)
                total.append(LUT[idx])
                # total.append(LUT[tup])
        return np.mean(total)

    def update(self, board, delta, alpha):
        # For each pattern, update all symmetric features equally.
        for pattern_index, syms in enumerate(self.symmetry_patterns):
            LUT = self.weights[pattern_index]
            for sym in syms:
                feature = self.get_feature(board, sym)
                idx = self.compute_index(feature)
                LUT[idx] += alpha * delta
                # LUT[feature] += alpha * delta

def peek_afterstate(env, action):
    temp_env = copy.deepcopy(env)
    temp_env.score = 0
    if action == 0:
        moved = temp_env.move_up()
    elif action == 1:
        moved = temp_env.move_down()
    elif action == 2:
        moved = temp_env.move_left()
    elif action == 3:
        moved = temp_env.move_right()

    afterstate = np.copy(temp_env.board)
    reward = temp_env.score
    return afterstate, reward

import copy
import random
import math
import numpy as np

# Example environment import; ensure your real environment is in scope
# from your_game_env import Game2048Env

class TD_MCTS_Node:
    def __init__(self, state, score, parent=None, action=None, is_random_node=False, exploration_constant=1.41):
        """
        state: current board state (numpy array)
        score: cumulative score
        parent: parent node (None for root)
        action: action taken from parent to this node (if applicable)
        is_random_node: if True, untried_actions will be random tile placements; 
                        if False, they will be player moves.
        exploration_constant: constant c for UCT calculation
        """
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.is_random_node = is_random_node
        self.c = exploration_constant
        self.children = {}
        self.visits = 0
        self.total_value = 0.0
        self.untried_actions = []
        
        if self.is_random_node:
            self.untried_actions = self.get_possible_tile_placements()
        else:
            self.untried_actions = [a for a in range(4) if self.is_move_legal(a)]

    def get_possible_tile_placements(self):
        empty_cells = list(zip(*np.where(self.state == 0)))
        placements = []
        for cell in empty_cells:
            placements.append((cell, 2))  # 90% chance
            placements.append((cell, 4))  # 10% chance
        return placements

    def is_move_legal(self, action):
        temp_env = Game2048Env()
        temp_env.board = self.state.copy()
        temp_env.score = self.score
        return temp_env.is_move_legal(action)

    def fully_expanded(self):
        return len(self.untried_actions) == 0

    def get_uct_value(self, parent_visits):
        """Compute the UCT value for this node."""
        if self.visits == 0:
            return float('inf')
        average_value = self.total_value / self.visits
        normalized_value = average_value/25000  # normalization factor (adjust as needed)
        exploration_term = self.c * math.sqrt(math.log(parent_visits) / self.visits)
        return normalized_value + exploration_term

class TD_MCTS:
    def __init__(self, env, approximator, iterations=500, exploration_constant=1.41, rollout_depth=10, gamma=0.99):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma

    def create_env_from_state(self, state, score):
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node):
        # If the node is a random node, select based on weighted tile placements.
        if node.is_random_node:
            placements = list(node.children.keys())
            weights = [0.9 if placement[1] == 2 else 0.1 for placement in placements]
            selected = random.choices(placements, weights=weights)[0]
            return node.children[selected]
        else:
            best_child = None
            best_value = -float('inf')
            for child in node.children.values():
                uct_value = child.get_uct_value(node.visits)
                if uct_value > best_value:
                    best_value = uct_value
                    best_child = child
            return best_child

    def expand(self, node):
        if node.is_random_node:
            # Expansion for random node: choose a tile placement.
            untried = node.untried_actions[:]
            weights = [0.9 if tile[1] == 2 else 0.1 for tile in untried]
            selected_tile = random.choices(untried, weights=weights)[0]
            
            (x, y), tile_value = selected_tile
            new_state = node.state.copy()
            new_state[x, y] = tile_value
            
            # Check for duplicate child states
            if any(np.array_equal(child.state, new_state) for child in node.children.values()):
                node.untried_actions.remove(selected_tile)
                # Optionally try another tile or return here.
            new_score = node.score
            # After random tile placement, node becomes an action node (is_random_node = False).
            child_node = TD_MCTS_Node(new_state, new_score, parent=node, action=None,
                                      is_random_node=False, exploration_constant=self.c)
            node.children[selected_tile] = child_node
            node.untried_actions.remove(selected_tile)
            return child_node
        else:
            action = random.choice(node.untried_actions)
            sim_env = self.create_env_from_state(node.state, node.score)
            afterstate, reward = peek_afterstate(sim_env, action)
            new_state = afterstate
            new_score = sim_env.score + reward
            child_node = TD_MCTS_Node(new_state, new_score, parent=node, action=action,
                                      is_random_node=True, exploration_constant=self.c)
            child_node.reward = reward  # store immediate reward if desired
            node.children[action] = child_node
            node.untried_actions.remove(action)
            return child_node

    def rollout(self, node, depth):
        """Perform a rollout and return a discounted cumulative reward estimate."""
        total_reward = 0
        discount = 1.0
        sim_env = self.create_env_from_state(node.state, node.score)

        for _ in range(depth):
            if sim_env.is_game_over():
                break
            legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
            if not legal_moves:
                break
            action = random.choice(legal_moves)
            old_score = sim_env.score
            _, _, done, _ = sim_env.step(action)
            step_reward = sim_env.score - old_score
            total_reward += discount * step_reward
            discount *= self.gamma
            if done:
                break

        # One-step lookahead: for remaining legal moves, estimate value 
        legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
        best_reward = float('-inf')
        for move in legal_moves:
            afterstate, reward = peek_afterstate(sim_env, move)
            value = self.approximator.value(afterstate) + reward
            if value > best_reward:
                best_reward = value
        return best_reward

    def backpropagate(self, node, reward):
        discount = 1.0
        # Propagate back up the tree with a decaying discount multiplier.
        while node is not None:
            node.visits += 1
            node.total_value += discount * reward
            if hasattr(node, 'reward'):
                reward += node.reward
            discount *= self.gamma
            node = node.parent

    def run_simulation(self, root):
        node = root
        # Selection: descend until a node is not fully expanded.
        while node.fully_expanded() and node.children:
            node = self.select_child(node)
        # Expansion: if node is not fully expanded, then expand it.
        if not node.fully_expanded():
            node = self.expand(node)
        # Rollout: if the node is a player move (non-random), you perform rollout;
        # for random nodes you might use the approximator directly.
        if node.is_random_node:
            rollout_value = self.approximator.value(node.state)
        else:
            rollout_value = self.rollout(node, self.rollout_depth)
        # Backpropagation: update all nodes along the simulation path.
        self.backpropagate(node, rollout_value)

    def best_action_distribution(self, root):
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action, distribution


approximator = NTupleApproximator(4, patterns)

with open('weights.pkl', 'rb') as f:
    approximator.weights = pickle.load(f)

def get_action(state, score):
    env = Game2048Env()
    env.board = state
    env.score = score
    td_mcts = TD_MCTS(env, approximator, iterations=100, exploration_constant=1.41, rollout_depth=0, gamma=0.99)
    root = TD_MCTS_Node(state, env.score)
    for _ in range(td_mcts.iterations):
        td_mcts.run_simulation(root)
    best_act, _ = td_mcts.best_action_distribution(root)
    return best_act # Choose a random action
    


