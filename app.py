from flask import Flask, render_template, request, jsonify
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

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/move", methods=["POST"])
def move():
    input_board = request.json["board"]
    # filter the board values 0 to -1(Empty positions), white = 1 to 0, black = -1 to 1.
    
    input_board = request.json["board"]

    for pos in range(len(input_board)):
        if input_board[pos] == 0:
            input_board[pos] = -1
        elif input_board[pos] == 1:
            input_board[pos] = 0
        elif input_board[pos] == -1:
            input_board[pos] = 1



    board_tensor = torch.tensor(input_board, dtype=torch.float32)
    with torch.no_grad():
        move = GoBot(board_tensor)

    move = move.argmax().item()
    print(f"Backend move: {move}")  # Add this line
    return jsonify({"move": move})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)