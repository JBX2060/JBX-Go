from sgfmill import sgf, boards, ascii_boards
from sgfmill.boards import Board
from sgfmill.ascii_boards import render_board
import numpy as np
import copy

Test = False


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
    
def convert_to_onehot(row, column):
    position = np.zeros(19 * 19)
    # row + column * (row length)
    index = row + column * 19
    
    position[index] = 1
    return position

