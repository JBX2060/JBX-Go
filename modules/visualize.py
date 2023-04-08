import numpy as np
from PIL import Image, ImageDraw

def create_go_board_image(board, file_path):
    # Define the colors for the stones and empty positions
    black_stone = (0, 0, 0)
    white_stone = (255, 255, 255)
    empty_pos = (222, 184, 135)
    
    # Create a blank image for the board
    img_size = board.shape[0] * 50
    img = Image.new('RGB', (img_size, img_size), (222, 184, 135))
    
    # Draw the lines for the board
    for i in range(board.shape[0]):
        x0, y0, x1, y1 = i*50, 0, i*50, img_size
        draw = ImageDraw.Draw(img)
        draw.line((x0, y0, x1, y1), fill=(0, 0, 0), width=2)
        draw.line((y0, x0, y1, x1), fill=(0, 0, 0), width=2)

    # Add the stones to the board
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if board[i, j] == 0:
                color = black_stone
            elif board[i, j] == 1:
                color = white_stone
            else:
                continue
                
            x, y = i*50, j*50
            draw = ImageDraw.Draw(img)
            draw.ellipse((x-25, y-25, x+25, y+25), fill=color, outline=(0, 0, 0), width=2)
    if file_path is not None:
        img.save(file_path)

    return img
