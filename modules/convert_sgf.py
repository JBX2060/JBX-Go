from sgfmill import sgf, boards, ascii_boards
# from modules.sgfmill2.boards import Board
from sgfmill.boards import Board
from sgfmill.ascii_boards import render_board
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from multiprocessing import Pool
from modules.visualize import create_go_board_image

import numpy as np
from tqdm import tqdm
import copy
import os

Test = True

def process_file(file_path):

    data_function = get_data

    game_boards, game_labels = data_function(file_path)
    formatted_boards_labels = [(format_board(board), label) for board, label in zip(game_boards, game_labels)]
    # python iter() method returns the iterator object, it is used to convert an iterable to the iterator.
    return iter(formatted_boards_labels)

def process_files_in_parallel(file_paths, max_workers=38):
    all_boards = []
    all_labels = []

    # print("running!")

    soft_run = False
    if soft_run:
        file_paths = file_paths[:4000]

    with Pool(processes=max_workers) as pool:
        results = pool.imap_unordered(process_file, file_paths)

        with tqdm(total=len(file_paths)) as pbar:
            for result in results:
                for formatted_board, game_label in result:
                    all_boards.append(formatted_board)
                    all_labels.append(game_label)
                    del formatted_board
                    del game_label
                pbar.update()

    return all_boards, all_labels

def get_data(file_path): 
    board = Board(19)

    board_list = [np.array(board.board)]
    labels = []

    with open(file_path, 'rb') as f:
        try:
            game = sgf.Sgf_game.from_bytes(f.read())
        except ValueError as e:
            # print(f"Error reading file {file_path}: {e}")
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

            # print(f"Invalid move in file {file_path} at move {i}, row: {row}, column: {column}, color: {color}")
            continue

        labels.append(row + column * 19)
        board_list.append(np.array(board.board))

    

    return board_list[:-1], labels

def format_board(board):
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

    board_arr = board_arr.astype(np.int8)

    return board_arr

def convert_to_onehot(row, column):
    position = np.zeros(19 * 19)
    # row + column * (row length)
    index = row + column * 19
    
    position[index] = 1
    return position

