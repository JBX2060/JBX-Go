from flask import Flask, render_template, request, jsonify, send_from_directory

import torch
from modules.model import Model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

board_size = 19

# Load the trained model
GoBot = Model()
GoBot.load_state_dict(torch.load("model_test.pth"))
GoBot.eval()

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory(app.static_folder, path)

def index_to_move(move):
    board_size = 19
    letters = "ABCDEFGHJKLMNOPQRST"
    
    row = move // board_size
    col = move % board_size
    move_str = f"{letters[col]}{board_size - row}"
    
    return move_str
    
def board_pos_to_coord(board_pos):
    x = (board_pos - 1) % 8
    y = (board_pos - 1) // 8
    return x, y

@app.route("/move", methods=["POST"])
def move():
    input_board = request.json["board"]


    board_tensor = torch.tensor(input_board, dtype=torch.int8)
    with torch.no_grad():
        move = GoBot(board_tensor)

    move = move.argmax().item()
    print(f"Backend move: {move}")  # Add this line

    move_str = index_to_move(move)    
    return jsonify({"move": move_str})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)