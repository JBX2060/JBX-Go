import numpy as np
import re
import os
import logging
import sgfmill
from sgfmill.boards import Board
import copy
from sgfmill import sgf, ascii_boards

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

processed_data_path = 'bot_data/kata2'
file_paths = [os.path.join(processed_data_path, file_name) for file_name in os.listdir(processed_data_path)]

class CustomGoBoard:
    def __init__(self, size=19):
        self.size = size
        self.board = np.full((size, size), -1, dtype=np.int8)
        self.previous_board = np.full((size, size), -1, dtype=np.int8)
        self.neighbors = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.history = []

    def is_legal_move(self, row, col, color):
        if not (0 <= row < self.size and 0 <= col < self.size):
            return False
        if self.board[row, col] != -1:
            return False

        self.board[row, col] = color
        if self.board.tolist() == self.previous_board.tolist():
            self.board[row, col] = -1
            return False

        opp_color = 1 - color
        captured = False
        for dr, dc in self.neighbors:
            r, c = row + dr, col + dc
            if 0 <= r < self.size and 0 <= c < self.size and self.board[r, c] == opp_color:
                if self.is_captured(r, c):
                    captured = True
                    self.remove_captured(r, c)

        self.board[row, col] = -1
        return captured or not self.is_captured(row, col)
    
    def _capture_stones(self, row, col, color):
        opponent_color = 1 - color
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dr, dc in directions:
            r, c = row + dr, col + dc
            if self.is_inside_board(r, c) and self.board[r, c] == opponent_color:
                captured_group, liberties = self._find_group_and_liberties(r, c)
                if len(liberties) == 0:
                    for r, c in captured_group:
                        self.board[r, c] = -1

    def _find_group_and_liberties(self, row, col):
        group = {(row, col)}
        group_color = self.board[row, col]
        liberties = set()
        visited = set()

        queue = [(row, col)]

        while queue:
            r, c = queue.pop(0)
            visited.add((r, c))

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc

                if self.is_inside_board(nr, nc) and (nr, nc) not in visited:
                    if self.board[nr, nc] == -1:
                        liberties.add((nr, nc))
                    elif self.board[nr, nc] == group_color:
                        group.add((nr, nc))
                        queue.append((nr, nc))

        return group, liberties
    
    def is_inside_board(self, row, col):
        return 0 <= row < self.size and 0 <= col < self.size

    def play_move(self, row, col, color, file_path=None):
        if self.is_legal_move(row, col, color):
            self.board[row, col] = color
            self.history.append((row, col))
            self._capture_stones(row, col, color)
        else:
            logging.warning(f"Illegal move in file {file_path} at move ({row}, {col}), color: {color}")




    def is_captured(self, row, col):
        color = self.board[row, col]
        visited = np.full((self.size, self.size), False, dtype=np.bool_)
        stack = [(row, col)]

        while stack:
            r, c = stack.pop()
            visited[r, c] = True
            for dr, dc in self.neighbors:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size and not visited[nr, nc]:
                    if self.board[nr, nc] == -1:
                        return False
                    elif self.board[nr, nc] == color:
                        stack.append((nr, nc))

        return True

    def remove_captured(self, row, col):
        color = self.board[row, col]
        stack = [(row, col)]

        while stack:
            r, c = stack.pop()
            self.board[r, c] = -1
            for dr, dc in self.neighbors:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size and self.board[nr, nc] == color:
                    stack.append((nr, nc))

def custom_sgf_parser(file_path):
    
    from sgfmill2.boards import Board

    board = Board(19)
    board_list = [copy.deepcopy(board)]
    labels = []

    with open(file_path, 'rb') as f:
        try:
            game = sgf.Sgf_game.from_bytes(f.read())
        except ValueError as e:
            print(f"Error reading file {file_path}: {e}")
            return [], []

    for i, node in enumerate(game.get_main_sequence()):
        move = node.get_move()

        if move[0] is None or move[1] is None:
            continue

        color = move[0]
        row = move[1][0]
        column = move[1][1]

        try:
            board.play(row, column, color)
        except ValueError:
            print(f"Invalid move in file {file_path} at move {i}, row: {row}, column: {column}, color: {color}")
            continue

        labels.append(row + column * 19)
        board_list.append(copy.deepcopy(board))
    

    return board_list[:-1], labels


def sgfmill_parser(file_path): 
    board = Board(19)
    board_list = [copy.deepcopy(board)]
    labels = []

    with open(file_path, 'rb') as f:
        try:
            game = sgf.Sgf_game.from_bytes(f.read())
        except ValueError as e:
            print(f"Error reading file {file_path}: {e}")
            return [], []

    for i, node in enumerate(game.get_main_sequence()):
        move = node.get_move()

        if move[0] is None or move[1] is None:
            continue

        color = move[0]
        row = move[1][0]
        column = move[1][1]

        try:
            board.play(row, column, color)
        except ValueError:
            print(f"Invalid move in file {file_path} at move {i}, row: {row}, column: {column}, color: {color}")
            continue

        labels.append(row + column * 19)
        board_list.append(copy.deepcopy(board))
    

    return board_list[:-1], labels



def test_parsers(file_paths, num_files_to_test=500):
    for i, file_path in enumerate(file_paths[:num_files_to_test]):
        print(f"Processing file {i + 1}: {file_path}")

        sgfmill_boards, sgfmill_labels = sgfmill_parser(file_path)
        custom_boards, custom_labels = custom_sgf_parser(file_path)

        if len(sgfmill_boards) != len(custom_boards) or len(sgfmill_labels) != len(custom_labels):
            print(f"Error: Mismatch in the number of boards or labels for file {file_path}")
            continue

        for j, (sgfmill_board, custom_board) in enumerate(zip(sgfmill_boards, custom_boards)):
            if not np.array_equal(sgfmill_board, custom_board):
                print(f"Error: Mismatch in board {j} for file {file_path}")

        for j, (sgfmill_label, custom_label) in enumerate(zip(sgfmill_labels, custom_labels)):
            if sgfmill_label != custom_label:
                print(f"Error: Mismatch in label {j} for file {file_path}")

    print("Test finished")

if __name__ == "__main__":
    test_parsers(file_paths)

# Replace `custom_sgf_parser` in the `process_file` function with the new version

   
