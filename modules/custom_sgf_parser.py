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

def format_board_test(board):
    # This function formats a board for further processing by converting its elements into numerical values.
    # The board is represented as a 3D numpy array of shape (17, 19, 19).

    # Create a new numpy array to hold the formatted board
    board_arr = np.array(board)

    # Iterate through the board and update the board_arr values
    white_mask = board_arr == 'w'
    black_mask = board_arr == 'b'
    empty_mask = ~(white_mask | black_mask)

    board_arr[black_mask] = 1
    board_arr[white_mask] = 0
    board_arr[empty_mask] = -1

    return board_arr


def sgfmill_parser(file_path): 
    board = Board(19)

    board_list = [np.array(board.board)]
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

        # board_list = 

        board_list.append(np.array(format_board_test(board.board)))

    

    return board_list[:-1], labels



def test_parsers(file_paths, num_files_to_test=25):
    for i, file_path in enumerate(file_paths[:num_files_to_test]):
        print(f"Processing file {i + 1}: {file_path}")

        sgfmill_boards, sgfmill_labels = sgfmill_parser(file_path)

    print("Test finished")

if __name__ == "__main__":
    test_parsers(file_paths)

# Replace `custom_sgf_parser` in the `process_file` function with the new version