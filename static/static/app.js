document.addEventListener("DOMContentLoaded", function () {
    var boardElement = document.getElementById("board");
    var board = new WGo.Board(boardElement, { size: 19 });
    var game = new WGo.Game();
  
    board.addEventListener("click", function (x, y) {
      if (game.isValid(x, y)) {
        var move = { x: x, y: y, c: game.turn };
        game.play(move);
        board.addObject({
          x: move.x,
          y: move.y,
          c: move.c,
          type: "STONE",
        });
  
        fetchMove(game.getBoard());
      }
    });
  
    function fetchMove(goBoard) {
      var boardArray = [];
      for (var x = 0; x < 19; x++) {
        for (var y = 0; y < 19; y++) {
          boardArray.push(goBoard.get(x, y));
        }
      }
  
      $.ajax({
        url: "/move",
        type: "POST",
        contentType: "application/json",
        data: JSON.stringify({ board: boardArray }),
        success: function (data) {
          var move = data.move;
          var x = move % 19;
          var y = Math.floor(move / 19);
  
          if (game.isValid(x, y)) {
            var botMove = { x: x, y: y, c: game.turn };
            game.play(botMove);
            board.addObject({
              x: botMove.x,
              y: botMove.y,
              c: botMove.c,
              type: "STONE",
            });
          }
        },
      });
    }
  });
  