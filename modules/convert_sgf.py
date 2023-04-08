from sgfmill import sgf, boards, ascii_boards
from sgfmill.boards import Board
from sgfmill.ascii_boards import render_board
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from multiprocessing import Pool
from visualize import create_go_board_image
import numpy as np
import copy
import os

Test = True

def process_file(file_path):

    data_function = get_data_test

    game_boards, game_labels = data_function(file_path)
    formatted_boards_labels = [(format_board_test(board), label) for board, label in zip(game_boards, game_labels)]
    # python iter() method returns the iterator object, it is used to convert an iterable to the iterator.
    return iter(formatted_boards_labels)

def process_files_in_parallel(file_paths, max_workers=16):
    all_boards = []
    all_labels = []

    # print("running!")

    soft_run = True
    if soft_run:
        file_paths = file_paths[:4000]

    with Pool(processes=max_workers) as pool:
        results = pool.imap_unordered(process_file, file_paths)

        for result in results:
            for formatted_board, game_label in result:
                #if formatted_board.size == 0 or not game_label:
                #    continue
                all_boards.append(formatted_board)
                all_labels.append(game_label)
                # print(f"Loading board from file: {file_paths}")
                
                # Free up memory, still does not work :(
                del formatted_board
                del game_label

    return all_boards, all_labels

def get_data_test(file_path): 
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

def get_data(file_path): 
    # This function reads a SGF (Smart Game Format) file and returns a list of boards
    # representing the states of the board after each move of the game.

    # Initialize a 19x19 Go board
    board = Board(19)

    # Copy the initial board to the list of boards
    board_list = [copy.deepcopy(board)]

    # Create labels for training, label = next move
    # Label format: (row, column, color)
    labels = []

    # Open the SGF file in binary mode and read its content
    with open(file_path, 'rb') as f:
        game = sgf.Sgf_game.from_bytes(f.read())

    # Loop through each node in the main sequence of the game
    for i, node in enumerate(game.get_main_sequence()):
        # Get the move made at this node
        move = node.get_move()  # (color (row, column))

        # Skip this node if no move was made (i.e. if the node has no color or coordinates)
        if move[0] is None or move[1] is None:
            continue

        # Extract the color and coordinates of the move
        color = move[0]
        row = move[1][0]
        column = move[1][1]

        # Play the move on the board
        board.play(row, column, color)
        labels.append(row + column * 19)

        # Copy the updated board to the list of boards
        board_list.append(copy.deepcopy(board))

        if Test==True and i > 4:
            break

    # Return the list of boards
    return board_list[:-1], labels


def format_board(board):
    # This function formats a board for further processing by converting its elements into numerical values.
    # The board is represented as a 1D numpy array.

    # Convert the board into a 1D numpy array
    board_arr = np.array(board.board)
    board_arr = board_arr.flatten()

    print("Formatting board: ", board)


    # Convert the elements of the array into numerical values
    for pos in range(len(board_arr)):
        if board_arr[pos] is None:
            board_arr[pos] = -1
        elif board_arr[pos] == 'w':
            board_arr[pos] = 0
        elif board_arr[pos] == 'b':
            board_arr[pos] = 1
    # Convert to floats
    board_arr = board_arr.astype(np.float32)


    # Return the formatted board
    return board_arr
    

def format_board_test(board):
    # This function formats a board for further processing by converting its elements into numerical values.
    # The board is represented as a 3D numpy array of shape (17, 19, 19).

    # Create a new numpy array to hold the formatted board
    board_arr = np.zeros((19, 19), dtype=np.float32)

    # Iterate through the board and update the board_arr values
    for i in range(19):
        for j in range(19):
            stone = board.get(i, j)
            if stone is None:
                board_arr[i, j] = -1
            elif stone == 'w':
                board_arr[i, j] = 0
            elif stone == 'b':
                board_arr[i, j] = 1

    # Create masks for each board element (None, 'w', 'b')
    none_mask = board_arr == -1
    w_mask = board_arr == 0
    b_mask = board_arr == 1

    # Fill the black_stones, white_stones, and empty_positions arrays using masks
    board_arr[b_mask] = 1
    board_arr[w_mask] = 0


    if os.path.exists('board_test.py'):
        print("Skipping checks!")
    else:
        create_go_board_image(board_arr, f'board_test.png')

    return board_arr





def convert_to_onehot(row, column):
    position = np.zeros(19 * 19)
    # row + column * (row length)
    index = row + column * 19
    
    position[index] = 1
    return position

