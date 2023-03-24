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
"""
Added batch normalization after each linear layer to help with faster convergence and better generalization.
Adjusted the number of hidden units for each layer to reduce the number of parameters and improve computation efficiency.
Added a dropout layer to reduce overfitting.
"""
class Model(nn.Module):
    def __init__(self):
        # Initialize the parent class
        super(Model, self).__init__()
        # Define the layers of the GoBot model
        # I have set the values into None for now, before I can figure 
        # out the correct values to use
        self.bot_layers = nn.Sequential(
            nn.Linear(361, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 361)
        )

    def forward(self, board):
        # self, input

        return self.bot_layers(board)
    
# Load model, if file exists
if os.path.exists("model.pth"):
    GoBot = Model()
    GoBot.load_state_dict(torch.load("model.pth"))
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
            create_go_board_image(board_batch[0].cpu().numpy(), f"images/epoch_{epoch_idx}_batch_{i}.png")


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
            torch.save(GoBot.state_dict(), "model.pth")

        except KeyboardInterrupt:
            print("Saving ...")
            torch.save(GoBot.state_dict(), "model.pth")
            sys.exit()
        
        # print("Correct: ", correct)
        # print("Wrong: ", wrong)

training_loop(10000)