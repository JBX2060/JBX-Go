import numpy as np
import torch

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.action = action
        self.N = 0
        self.W = 0
        self.Q = 0
        self.P = 0

class MCTS:
    def __init__(self, model, game, num_simulations=100, cpuct=1):
        self.model = model
        self.game = game
        self.num_simulations = num_simulations
        self.cpuct = cpuct

    def search(self, state):
        root = Node(state)

        for _ in range(self.num_simulations):
            node = self.select(root)
            if not self.game.is_terminal(node.state):
                node = self.expand(node)
                reward = self.simulate(node)
            else:
                reward = self.game.get_reward(node.state)
            self.backpropagate(node, reward)

        return self.get_best_action(root)

    def select(self, node):
        while node.children:
            node = max(node.children, key=lambda child: child.Q + self.cpuct * child.P * np.sqrt(node.N) / (1 + child.N))
        return node

    def expand(self, node):
        actions = self.game.get_valid_actions(node.state)
        with torch.no_grad():
            input_state = self.game.state_to_input(node.state)
            action_probs = self.model(torch.tensor(input_state).float().unsqueeze(0)).detach().numpy().flatten()
        for action in actions:
            child_state = self.game.step(node.state, action)
            child = Node(child_state, parent=node, action=action)
            child.P = action_probs[action]
            node.children.append(child)
        return node.children[np.argmax(action_probs[actions])]

    def simulate(self, node):
        state = node.state
        while not self.game.is_terminal(state):
            action = np.random.choice(self.game.get_valid_actions(state))
            state = self.game.step(state, action)
        return self.game.get_reward(state)

    def backpropagate(self, node, reward):
        while node is not None:
            node.N += 1
            node.W += reward
            node.Q = node.W / node.N
            node = node.parent

    def get_best_action(self, root):
        return max(root.children, key=lambda child: child.N).action

class GoGame:
    def __init__(self, board_size=19):
        self.board_size = board_size

    def get_valid_actions(self, state):
        # Get the current player color (1 for black, -1 for white)
        player = state[0]

        # Get the board state (binary matrix with 1 for stones and 0 for empty cells)
        board = state[1:]

        # Find all empty cells on the board
        empty_cells = np.where(board == -1)

        # Initialize a list to store valid actions
        valid_actions = []

        # Check each empty cell for validity
        for i, j in zip(empty_cells[0], empty_cells[1]):
            # Check if the move is legal (i.e., does not violate any of the game rules)
            if self.is_legal_move(board, player, i, j):
                # If the move is legal, add it to the list of valid actions
                valid_actions.append((i, j))

        return valid_actions

    def step(self, state, action):
        # Get the current player color
        player = state[0]

        # Get the board state
        board = state[1:]

        # Place a stone of the current player color at the specified position
        board[action] = player

        # Switch the player color for the next turn
        next_player = -player

        # Return the new state
        return (next_player, board)

    def is_terminal(self, state):
        # Get the board state
        board = state[1:]

        # Check if the board is full (i.e., no more empty cells)
        if np.count_nonzero(board == -1) == 0:
            return True

        # Check if either player has captured all the stones of the other player
        black_stones = np.sum(board == 0)
        white_stones = np.sum(board == 1)
        if black_stones == 0 or white_stones == 0:
            return True

        # Check if both players passed their turn
        if state[2] and state[3]:
            return True

        # If none of the above conditions are met, the game is not over yet
        return False

    def get_reward(self, state):
        # Get the board state
        board = state[1:]

        # Check if the game is over
        if self.is_terminal(state):
            # Compute the final score (i.e., the difference between the number of black stones and the number of white stones)
            black_stones = np.sum(board == 0)
            white_stones = np.sum(board == 1)
            score = black_stones - white_stones

            # Return the reward based on the final score
            if score > 0:
                return 1.0  # Black wins
            elif score < 0:
                return -1.0  # White wins
            else:
                return 0.0  # Draw

        # If the game is not over yet, return 0 as the reward (i.e., the game is still in progress)
        return 0.0
    
    def state_to_input(self, state):
        # Get the board state
        board = state[1:]

        # Convert the board to a tensor with shape (1, board_size, board_size)
        board_tensor = np.expand_dims(board, axis=0)

        # Convert the tensor to the input format required by the model
        # (i.e., a tuple of two tensors, one for black stones and one for white stones)
        black_stones = np.zeros((1, self.board_size, self.board_size))
        black_stones[board == 1] = 1.0
        white_stones = np.zeros((1, self.board_size, self.board_size))
        white_stones[board == -1] = 1.0
        return (black_stones, white_stones)
    
    def is_legal_move(self, board, player, i, j):
        # Check if the specified cell is empty
        if board[i, j] != 0:
            return False

        # Check if the move would result in a self-capture (i.e., the player's own stones would be surrounded)
        temp_board = np.copy(board)
        temp_board[i, j] = player
        if self.is_self_capture(temp_board, player, i, j):
            return False

        # Check if the move would result in a capture of the opponent's stones
        if self.is_capture(board, player, i, j):
            return True

        # Check if the move would connect the player's stones to an existing group
        if self.is_connection(board, player, i, j):
            return True

        # If none of the above conditions are met, the move is illegal
        return False

    def is_self_capture(self, board, player, i, j):
        # Check if any adjacent opponent stones are captured
        for x, y in self.get_adjacent_cells(i, j):
            if board[x, y] == -player:
                if self.is_capture(board, -player, x, y):
                    return False

        # Check if the group of the specified stone is surrounded by opponent stones
        visited = np.zeros_like(board)
        group = self.get_group(board, player, i, j, visited)
        for x, y in group:
            for u, v in self.get_adjacent_cells(x, y):
                if board[u, v] == 0:
                    return False
                elif board[u, v] == player and visited[u, v] == 0:
                    if len(self.get_group(board, player, u, v, visited)) > 1:
                        return False

        # If none of the above conditions are met, the group is self-captured
        return True

    def is_capture(self, board, player, i, j):
        # Check if any adjacent opponent groups are surrounded and captured
        for x, y in self.get_adjacent_cells(i, j):
            if board[x, y] == -player:
                visited = np.zeros_like(board)
                group = self.get_group(board, -player, x, y, visited)
                if self.is_surrounded(board, group):
                    return True

        # If none of the adjacent opponent groups are surrounded, no capture is made
        return False

    def is_connection(self, board, player, i, j):
        # Check if any adjacent groups of the same color are connected
        for x, y in self.get_adjacent_cells(i, j):
            if board[x, y] == player:
                visited = np.zeros_like(board)
                group = self.get_group(board, player, x, y, visited)
                if len(group) > 1:
                    return True

        # If none of the adjacent groups are connected, no connection is made
        return False

    def get_adjacent_cells(self, i, j):
        # Return a list of adjacent cells (up, down, left, right)
        adjacent_cells = []
        if i > 0:   
            adjacent_cells.append((i - 1, j))
        if i < self.board_size - 1:
            adjacent_cells.append((i + 1, j))
        if j > 0:
            adjacent_cells.append((i, j - 1))
        if j < self.board_size - 1:
            adjacent_cells.append((i, j + 1))
        return adjacent_cells

    def get_group(self, board, player, i, j, visited):
        # Return a list of all stones in the group containing the specified cell
        group = [(i, j)]
        visited[i, j] = 1
        for x, y in self.get_adjacent_cells(i, j):
            if board[x, y] == player and visited[x, y] == 0:
                group += self.get_group(board, player, x, y, visited)
        return group

    def is_surrounded(self, board, group):
        # Check if the specified group is surrounded by opponent stones
        for i, j in group:
            for x, y in self.get_adjacent_cells(i, j):
                if board[x, y] == 0:
                    return False
                elif board[x, y] == group[0][2]:
                    continue
                elif board[x, y] == -group[0][2]:
                    if len(self.get_group(board, -group[0][2], x, y, np.zeros_like(board))) == 1:
                        return False
        return True


