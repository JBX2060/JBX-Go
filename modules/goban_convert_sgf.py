import re
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from custom_sgf_parser import custom_sgf_parser
from custom_sgf_parser import test_parsers

import cProfile
import os


# 'boards_extra_full', 'labels_extra_full', 'boards_full.npy', 'labels_full.npy', 'boards_large.npy', 'labels_large.npy', 'boards.npy', 'labels.npy'
processed_data_path = 'bot_data/kata2'

boards_path = '/home/jbx2060/JBX2020/proccesed_data/kata2_boards.npy'
labels_path = '/home/jbx2060/JBX2020/proccesed_data/kata2_labels.npy'

class SGFParser:
    def __init__(self, sgf_string):
        self.sgf_string = sgf_string
        self.moves = self.parse_moves()

    def parse_moves(self):
        moves = []
        for match in re.finditer(r";([BW])\[(\w\w)\]", self.sgf_string):
            color = 1 if match.group(1) == "B" else 0
            move = match.group(2)
            row = ord(move[0]) - ord("a")
            column = ord(move[1]) - ord("a")
            moves.append((color, row, column))
        return moves

def get_data_test(file_path):
    game_positions = []
    labels = []

    with open(file_path, 'r') as f:
        try:
            sgf_content = f.read()
            sgf_parser = SGFParser(sgf_content)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return np.empty((0, 19, 19)), []

    board = np.zeros((19, 19), dtype=np.float32)
    for color, row, column in sgf_parser.moves:
        if 0 <= row < 19 and 0 <= column < 19:
            board[row, column] = color
            labels.append(row + column * 19)
            game_positions.append(board.copy())
        else:
            print(f"Invalid move in file {file_path}, row: {row}, column: {column}, color: {color}")

    return np.array(game_positions), labels

def format_board(board):
    return board

def process_file(file_path):
    try:
        data_function = custom_sgf_parser

        game_boards, game_labels = data_function(file_path)
        formatted_boards_labels = [(format_board(board), label) for board, label in zip(game_boards, game_labels)]
        return iter(formatted_boards_labels)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return iter([])

def process_files_in_parallel(file_paths, max_workers=16):
    all_boards = []
    all_labels = []

    # print("running!")

    soft_run = False
    if soft_run:
        file_paths = file_paths[:4000]

    with Pool(processes=max_workers) as pool:
        results = pool.imap_unordered(process_file, file_paths)

        with tqdm(total=len(file_paths), desc="Processing files", unit="file", ncols=100) as pbar:
            for result in results:
                for formatted_board, game_label in result:
                    all_boards.append(formatted_board)
                    all_labels.append(game_label)
                    del formatted_board
                    del game_label
                pbar.update()

    return all_boards, all_labels


def load_data(boards_path=boards_path, labels_path=labels_path, processed_data_path=processed_data_path):
    if os.path.exists(boards_path) and os.path.exists(labels_path):
        print("Loading Training Dataset...")
        boards = np.load(boards_path)
        labels = np.load(labels_path
    )

    else:
        print("Processing data...")
        file_paths = [os.path.join(processed_data_path, file_name) for file_name in os.listdir(processed_data_path)]
        boards, labels = process_files_in_parallel(file_paths)

        np.save(boards_path, boards)
        np.save(labels_path, labels)

    return boards, labels

if __name__ == "__main__":
    cProfile.run('load_data()', 'profile_stats')