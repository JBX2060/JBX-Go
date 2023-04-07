# Standard library imports
import os
import sys

# Third-party imports
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms

# Local application imports
from convert_sgf import process_files_in_parallel
from visualize import create_go_board_image
from early_stopping import EarlyStopper


# Global Varibles
loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.0001
batch_size = 200

boards = []
validation_data = []
labels = []
use_cuda = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# Divide the dataset, folder=go_data/9d, into training and validation sets
# Training set: 80%
# Validation set: 20%
# Check if the processed boards and labels are already saved
# if __name__ == '__main__':
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

# Split the data into training and validation sets (80% training, 20% validation)
train_data, validation_data = train_test_split(data, test_size=0.2, random_state=42)

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

train_loader = DataLoader(list(zip(train_boards,train_labels)), shuffle=True, batch_size=200)
test_loader = DataLoader(list(zip(validation_boards,validation_labels)), shuffle=True, batch_size=200)

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
        # Used to bypass convolutional layers if they are not usefull...
        out += identity
        out = self.relu(out)
        return out

class Model(nn.Module):
    def __init__(self, num_res_blocks=18, channels=128):
        super(Model, self).__init__()
        self.conv_input = nn.Conv2d(3, channels, kernel_size=3, padding=1)

        self.bn_input = nn.BatchNorm2d(channels)

        # inplace is optional, but it can save some memory. It avoids a new tensor.
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
        # flatten the output of the convolutional layer
        policy = policy.view(policy.size(0), -1)
        # apply the fully connected layer to the output of the convolutional layer
        policy = self.fc_policy(policy)

        return policy
    
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

    for epoch_idx in range(n_epochs):

        correct = 0
        wrong = 0
        total_loss = 0
        test_total_loss = 0
        test_correct = 0
        test_wrong = 0

        print(f"EPOCH {epoch_idx}")

        for board_batch, (inputs, labels) in enumerate(train_loader):

            board_batch, labels_batch = inputs.to(device), labels.to(device)
            
            # logits -> probabilities via softmax
            # [5.235, -0.0323, 0.2] -> [0.75, 0.1, 0.15]

            # Train the model
            output = GoBot(board_batch)

            board_batch = board_batch.reshape(-1, 19, 19)

            if (board_batch % 50 == 0).all():
                create_go_board_image(board_batch[0].cpu().numpy(), f"images/epoch_{epoch_idx}_batch_{board_batch}.png")


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

        print(f"Evaluating epoch: {epoch_idx}")

        for test_batch, (inputs, labels) in enumerate(test_loader):

            board_batch, labels_batch = inputs.to(device), labels.to(device)
            # Validation
            with torch.no_grad():
                validation_output = GoBot(board_batch)
            
            test_loss = loss_fn(validation_output, labels_batch)
            test_total_loss = test_total_loss + test_total_loss

            validation_output = validation_output.argmax(dim=1)

            correct_labels = validation_output == labels_batch
            test_correct += correct_labels.sum()
            test_wrong += len(labels_batch) - correct_labels.sum()

        
        total_times = correct + wrong
        total_test_times = test_correct + test_wrong

        accuracy = correct / total_times * 100
        test_accuracy = test_correct / total_test_times * 100

        print("Accuracy: ", accuracy)
        print("Test Accuracy: ", test_accuracy)

        print("Avg Loss: ", total_loss / total_times)
        print(f"Avg Test Loss: {test_total_loss / total_test_times}")

        early_stopper = EarlyStopper(patience=3, min_delta=10)
        
        if early_stopper.early_stop(test_total_loss / total_test_times):
            print("We are at epoch:", epoch_idx)             
            break
            

        try:
            torch.save(GoBot.state_dict(), "model_test.pth")

        except KeyboardInterrupt:
            print("Saving ...")
            torch.save(GoBot.state_dict(), "model_test.pth")
            sys.exit()
        
        # print("Correct: ", correct)
        # print("Wrong: ", wrong)
# import nueral_search

if __name__ == '__main__':
    training_loop(10000)