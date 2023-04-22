import os

import torch
from torch import nn


def flat_to_channels(board):
    black = (board == 0).float()
    white = (board == 1).float()
    empty = (board == -1).float()
    return torch.stack([black, white, empty], dim=-1).view(-1, 3, 19, 19)


"""
Conv (Convolutional Layer):
A convolutional layer is a type of layer in a neural network specifically designed to process grid-like data, such as images. 
It uses small filters (also called kernels) that slide across the input data, detecting features like edges, shapes, and textures. 
It's particularly useful for tasks like image recognition and game board analysis, as it can capture spatial information more effectively than fully connected layers.


Batch Normalization (bn2 represents the second batch normalization in the ResidualBlock):
Batch normalization is a technique used to improve the training speed and stability of neural networks. 
It normalizes the activations (outputs) of a layer by adjusting and scaling them during training, making the learning process more efficient. 
It also helps in reducing the risk of vanishing or exploding gradients. 
In the code, 'bn2' represents the second batch normalization layer in the ResidualBlock.

ResidualBlock:
A Residual Block is a building block of a deep neural network. 
It consists of a few layers of Convolutional Neural Networks (CNNs) with a special connection called a "skip connection" or "shortcut connection". 
This connection allows the input to the block to be added directly to its output, which helps the network learn more effectively and reduces the risk of vanishing gradients when training deep networks. 
Think of it like taking a shortcut through layers, allowing the model to remember important information.
"""

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        # Kernal Size: size of convolution filter is matrix.
        # Padding: 1 means that the input is padded with a border of 1 pixel, so that the output has the same size as the input.
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
        # shortcut for model, redisual connection
        # Used to bypass convolutional layers if they are not useful...
        out += identity
        out = self.relu(out)
        return out

class Model(nn.Module):

    def __init__(self, num_res_blocks=6, channels=128, dropout_rate=0.5):

        super(Model, self).__init__()

        self.conv_input = nn.Conv2d(3, channels, kernel_size=3, padding=1)

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        self.bn_input = nn.BatchNorm2d(channels)

        # inplace is optional, but it can save some memory. It avoids a new tensor.
        self.relu = nn.ReLU(inplace=True)

        self.res_blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_res_blocks)])

        self.conv_policy = nn.Conv2d(channels, 2, kernel_size=1)
        self.bn_policy = nn.BatchNorm2d(2)
        self.relu_policy = nn.ReLU(inplace=True)
        self.fc_policy = nn.Linear(2 * 19 * 19, 361)

        # self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, board):
        # Convert the flat board representation to a 3-channel representation
        board = flat_to_channels(board)
        x = self.conv_input(board)

        # Pass though the residual blocks, and different layers
        x = self.bn_input(x)
        x = self.relu(x)
        x = self.res_blocks(x)
        # policy: probability distribution over the available actions

        # extract more abstract and high-level features from the input
        policy = self.conv_policy(x)
        policy = self.bn_policy(policy)
        
        # apply the ReLU activation function to the output of the batch normalization layer
        policy = self.relu_policy(policy)

        # Add dropout to the output of the ReLU activation function
        policy = self.dropout(policy) 

        # flatten the output of the convolutional layer
        policy = policy.view(policy.size(0), -1)
        # apply the fully connected layer to the output of the convolutional layer
        policy = self.fc_policy(policy)
        # policy = self.softmax(policy)

        return policy


def load_model(model_path, device):
    if os.path.exists(model_path):
        GoBot = Model()
        GoBot.load_state_dict(torch.load(model_path))
    else:
        GoBot = Model()

    GoBot.to(device)
    return GoBot
