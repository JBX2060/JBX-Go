from flask import Flask, render_template, request, jsonify
import torch
import argparse
import numpy as np
import json
import ast

from modules.model import Model
from flask_cors import CORS

import requests
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

app = Flask(__name__)
CORS(app)

board_size = 19

# Parse command-line arguments
parser = argparse.ArgumentParser(description="GoBot Flask Web Application")
parser.add_argument("--train", action="store_true", default=False, help="Train the model using the provided API")
args = parser.parse_args()

# Set a global variable to control whether the model should be trained
train_model_flag = args.train

# Load the trained model
GoBot = Model()
GoBot.load_state_dict(torch.load("model_test.pth"))

# Define a loss function and optimizer
loss_function = CrossEntropyLoss()
optimizer = Adam(GoBot.parameters(), lr=0.001)

# Convert the format of the board to the format expected by the API
def convert_board(input_board):
    print([-1 if pos == 0 else 0 if pos == 1 else 1 for pos in input_board])

    return [-1 if pos == 0 else 0 if pos == 1 else 1 for pos in input_board]

def board_to_moves(board):
    moves = []
    board_size = int(len(board)**0.5)
    for index, cell in enumerate(board):
        if cell != -1:
            i = index // board_size
            j = index % board_size
            column = chr(i + ord("A"))
            if column >= "I":
                column = chr(i + ord("A") + 1)  # Skipping the letter 'I' as is standard in Go notation
            row_number = board_size - j
            move = column + str(row_number)
            moves.append(move)
    json_moves = json.dumps(moves)
    print(json_moves)
    return json.dumps(moves)

# def board_to_moves(board):
#     moves = []
#     board_size = int(len(board)**0.5)
#     for index, cell in enumerate(board):
#         if cell != -1:
#             i = index // board_size
#             j = index % board_size
#             column = chr(j + ord('A'))
#             if column >= 'I':
#                 column = chr(j + ord('A') + 1)  # Skipping the letter 'I' as is standard in Go notation
#             row_number = board_size - i
#             move = f"{column}{row_number}"
#             if cell == 0:
#                 move = "W" + move
#             else:
#                 move = "B" + move
#             moves.append(move)
#     json_moves = json.dumps(moves)
#     print(json_moves)
#     return json.dumps(moves)


# Function to fetch the best move from the API
def get_best_move(board_size, moves):
    url = "http://127.0.0.1:8080/select-move/katago_gtp_bot"
    headers = {"Content-Type": "application/json"}
    
    # Convert the moves string to a list of strings without double quotes
    moves_list = ast.literal_eval(moves)
    moves_formatted = [move.replace("'", '"') for move in moves_list]
    # [["Q16"], ["Q4"]]
    # ["Q16", "Q4"]


    data = {
        "board_size": board_size,
        # Input for API: ["A1", "B2", "C3", "D4"]
        # Format I have: ['A1', 'B2', 'C3', 'D4']
        "moves": moves_formatted
    }

    print(data)
    print(json.dumps(data))
    response = requests.post(url, data=json.dumps(data), headers=headers)

    print(response.json())

    if response.status_code == 200:
        move_data = response.json()
        best_move = move_data["bot_move"]
        return best_move
    else:
        raise Exception(f"API call failed with status code {response.status_code}")

def move_to_indices(move, board_size=19):
    col, row = move[0], int(move[1:])
    col_index = ord(col.upper()) - ord('A')
    if col_index >= ord('I') - ord('A'):
        col_index -= 1
    row_index = board_size - row
    return row_index, col_index

def one_hot_move(move, board_size=19):
    row_index, col_index = move_to_indices(move, board_size)
    flat_index = row_index * board_size + col_index
    one_hot = torch.zeros(1, board_size * board_size, dtype=torch.float32)
    one_hot[0, flat_index] = 1.0
    return one_hot

# Function to train the model based on the API's best move
def train_model(input_board, best_move):
    # Preprocess the input_board and best_move as needed, e.g., convert to tensors

    
    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    output = GoBot(input_board)

    # Convert data
    best_move = one_hot_move(best_move)
    best_move = torch.tensor(best_move)

    print(best_move)

    # Calculate the loss
    loss = loss_function(output, best_move)

    # Backward pass
    loss.backward()

    # Update the model parameters
    optimizer.step()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/move", methods=["POST"])
def move():
    input_board = request.json["board"]
    input_board = convert_board(input_board)

    moves = board_to_moves(input_board)
    board_tensor = torch.tensor(input_board, dtype=torch.float32)


    if train_model_flag:
        # Get the best move from the API
        best_move = get_best_move(board_size, moves)
        print("Best move: ", best_move)
        if best_move:
            # Train the model using the best move from the API
            train_model(board_tensor, best_move)
        else:
            print("Error: Could not get the best move from the API")

    # Evaluate the model to make a move
    GoBot.eval()
    with torch.no_grad():
        move = GoBot(board_tensor)

    if train_model_flag:
        GoBot.train()



    # move = get_best_move(board_size, moves)

    # # Convert data

    # move = one_hot_move(move)
    # move = torch.tensor(move)
    # move = move.argmax().item()

    print(f"Backend move: {move}")
    return jsonify({"move": move})

if __name__ == "__main__":

    app.run(debug=True, host='0.0.0.0', port=8000)
