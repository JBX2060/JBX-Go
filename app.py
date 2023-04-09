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
    board_tensor = torch.tensor(input_board, dtype=torch.float32)
    with torch.no_grad():
        move = GoBot(board_tensor)

    move = move.argmax().item()
    return jsonify({"move": move})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)
