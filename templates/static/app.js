$(document).ready(function () {
    const boardSize = 19;

    const board = new WGo.Board(document.getElementById("board"), {
        width: 760,
        size: boardSize,
    });

    board.addEventListener("click", function (x, y) {
        if (board.obj_arr[x][y].length === 0) {
            board.addObject({
                x: x,
                y: y,
                type: WGo.Board.drawHandlers.GO,
                c: WGo.B,
            });

            // Send the board state to the server to get the bot's move
            makeMove(board.obj_arr);
        }
    });

    function makeMove(boardState) {
        let flatBoard = [];
        for (let x = 0; x < boardSize; x++) {
            for (let y = 0; y < boardSize; y++) {
                const obj = boardState[x][y];
                flatBoard.push(obj.length === 0 ? -1 : obj[0].c === WGo.B ? 0 : 1);
            }
        }

        $.ajax({
            type: "POST",
            url: "/move",
            contentType: "application/json",
            data: JSON.stringify({ board: flatBoard }),
            success: function (response) {
                const move = response.move;
                const x = move % boardSize;
                const y = Math.floor(move / boardSize);

                if (board.obj_arr[x][y].length === 0) {
                    board.addObject({
                        x: x,
                        y: y,
                        type: WGo.Board.drawHandlers.GO,
                        c: WGo.W,
                    });
                }
            },
        });
    }
});
