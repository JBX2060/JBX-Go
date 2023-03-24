from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import torch
from torch import nn
import sys
import os

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class Model(nn.Module):
    def __init__(self, num_res_blocks=6, channels=128):
        super(Model, self).__init__()
        self.conv_input = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        self.res_blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_res_blocks)])

        self.conv_policy = nn.Conv2d(channels, 2, kernel_size=1)
        self.bn_policy = nn.BatchNorm2d(2)
        self.relu_policy = nn.ReLU(inplace=True)
        self.fc_policy = nn.Linear(2 * 19 * 19, 361)

    def forward(self, board):
        # Convert the flat board representation to a 3-channel representation
        board = flat_to_channels(board)
        x = self.conv_input(board)

        x = self.bn_input(x)
        x = self.relu(x)
        x = self.res_blocks(x)

        policy = self.conv_policy(x)
        policy = self.bn_policy(policy)
        policy = self.relu_policy(policy)
        policy = policy.view(policy.size(0), -1)
        policy = self.fc_policy(policy)

        return policy
    
    def make_move(self, board, player_color):
        # Ensure the input board is a PyTorch tensor with the correct device
        board_tensor = torch.tensor(board, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            output = self(board_tensor)
        
        # Find the index with the highest probability
        move_index = output.argmax().item()

        # Create a one-hot encoded move
        move_one_hot = np.zeros(361, dtype=int)
        move_one_hot[move_index] = 1

        return move_one_hot

go_bot = Model()
go_bot.load_state_dict(torch.load("model_test.pth", map_location=device))
go_bot.to(device)


app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template(	'index.html')


# Initialize your Go bot here
# go_bot = YourGoBot()

def board_to_str(board):
    return '\n'.join([' '.join([str(cell) for cell in row]) for row in board])

def str_to_board(board_str):
    return np.array([[int(cell) for cell in row.split()] for row in board_str.split('\n')])

@app.route('/api/make_move', methods=['POST'])
def make_move():
    board_str = request.form['board']
    player_color = int(request.form['player'])

    board = str_to_board(board_str)

    # Make the bot's move
    one_hot_response = go_bot.make_move(board, player_color)

    bot_move_position = one_hot_to_position(one_hot_response)
    new_board = np.copy(board)
    new_board[bot_move_position] = player_color

    return jsonify(board=board_to_str(new_board))


#if __name__ == '__main__':
app.run(debug=True, port=8080)
