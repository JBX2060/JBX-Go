def board_to_moves(board):
    moves = []
    for i, row in enumerate(board):
        for j, cell in enumerate(row):
            if cell != -1:
                column = chr(j + ord('A'))
                if column >= 'I':
                    column = chr(j + ord('A') + 1)  # Skipping the letter 'I' as is standard in Go notation
                row_number = len(board) - i
                move = f"{column}{row_number}"
                if cell == 0:
                    move = "W" + move
                else:
                    move = "B" + move
                moves.append(move)
    return moves

board = [-1, -1, -1],
[-1,  0, -1],
[-1, -1, -1]
print(board_to_moves(board))