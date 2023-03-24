import os
import torch
import numpy as np
import sys

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from convert_sgf import process_files_in_parallel
from sklearn.model_selection import train_test_split
from visualize import create_go_board_image

# Global Varibles
loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.0001
batch_size = 200

boards = []
validation_data = []
labels = []
use_cuda = False

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")



# Divide the dataset, folder=go_data/9d, into training and validation sets
# Training set: 80%
# Validation set: 20%
# Check if the processed boards and labels are already saved
if os.path.exists("boards.npy") and os.path.exists("labels.npy"):
    # Load the processed boards and labels
    print("Loading Training Dataset...")
    boards = np.load("boards.npy")
    labels = np.load("labels.npy")
else:
    # Process the SGF files and save the processed boards and labels
    print("Processing data...")
    file_paths = [os.path.join("go_data/9d", file_name) for file_name in os.listdir("go_data/9d")]
    boards, labels = process_files_in_parallel(file_paths)
    
    
    # Save the processed boards and labels
    np.save("boards.npy", boards)
    np.save("labels.npy", labels)



# Combine boards and labels
data = list(zip(boards, labels))

# Shuffle the data
np.random.shuffle(data)

# Split the data into training and validation sets (80% training, 20% validation)
train_data, validation_data = train_test_split(data, test_size=0.2)

# Separate the boards and labels in the training and validation sets
train_boards, train_labels = zip(*train_data)
validation_boards, validation_labels = zip(*validation_data)

#
train_boards = np.stack(train_boards)
train_labels = np.stack(train_labels)
validation_boards = np.stack(validation_boards)
validation_labels = np.stack(validation_labels)

train_boards = torch.from_numpy(train_boards)
train_labels = torch.from_numpy(train_labels)
validation_boards = torch.from_numpy(validation_boards)
validation_labels = torch.from_numpy(validation_labels)


train_boards = train_boards.to(device)
train_labels = train_labels.to(device)
validation_boards = validation_boards.to(device)
validation_labels = validation_labels.to(device)

# Convert the data into tensors
boards = np.stack(boards)
labels = np.stack(labels)

boards = torch.from_numpy(boards)
labels = torch.from_numpy(labels)

boards = boards.reshape(len(boards), 361)

boards = boards.to(device)
labels = labels.to(device)


# print(boards[1])

# torch.tensor(validation_data)

def check_data():
    # Compare labels length and board length
    print(len(labels))

    print("Labels shape: ", labels.shape)
    print("Board shape: ", boards.shape)

check_data()


# Path: model.py

# Define the model,
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
    
# Load model, if file exists
if os.path.exists("model_test.pth"):
    GoBot = Model()
    GoBot.load_state_dict(torch.load("model_test.pth"))
else:
    GoBot = Model()

GoBot.to(device)

print("Using ", device)

optim = torch.optim.Adam(GoBot.parameters(), learning_rate)

def training_loop(n_epochs):
    # My plan is to loop through the training data files. Train the model on ONE file at a time. 
    # After a complete loop that would be one epoch.
    # Then I would loop through the validation data files. To determine the accuracy of the model.
    # I would then save the model if the accuracy is better than the previous model.
    # Loop through files


    for epoch_idx in range(n_epochs):
        correct = 0
        wrong = 0
        total_loss = 0
        print(f"EPOCH {epoch_idx}")

        for i in range(0, len(boards), batch_size):
            # label = labels[i]

            labels_batch = labels[i:i+batch_size]
            board_batch = boards[i:i+batch_size]
            
            # logits -> probabilities via softmax
            # [5.235, -0.0323, 0.2] -> [0.75, 0.1, 0.15]

            # Train the model
            output = GoBot(board_batch)

            board_batch = board_batch.reshape(-1, 19, 19)
            # create_go_board_image(board_batch[0].cpu().numpy(), f"images/epoch_{epoch_idx}_batch_{i}.png")


            loss = loss_fn(output, labels_batch)
            total_loss = total_loss + loss
            
            # Optimize the gradients
            optim.zero_grad()        
            loss.backward()
            optim.step()
            for j in range(len(labels_batch)):

                output_index = output[j].argmax()
                current_label = labels_batch[j]
                
                if output_index == current_label:
                    correct += 1
                else:
                    wrong += 1
            
            
        total_times = correct + wrong

        accuracy = correct / total_times * 100
        print("Accuracy: ", accuracy)
        print("Avg Loss: ", total_loss / total_times)

        try:
            torch.save(GoBot.state_dict(), "model_test.pth")

        except KeyboardInterrupt:
            print("Saving ...")
            torch.save(GoBot.state_dict(), "model_test.pth")
            sys.exit()
        
        # print("Correct: ", correct)
        # print("Wrong: ", wrong)
        
if __name__ == '__main__':
    training_loop(10000)