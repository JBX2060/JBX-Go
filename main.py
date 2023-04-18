import argparse
import torch
from torch import nn
from modules.data_preparation import load_data, prepare_data
from modules.model import Model, load_model
from modules.train import training_loop
from torch.utils.data import DataLoader

def main(num_epochs):
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 0.0001
    batch_size = 1150

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using ", device)

    boards, labels = load_data()
    train_boards, train_labels, validation_boards, validation_labels = prepare_data(boards, labels)

    train_boards = torch.from_numpy(train_boards)
    train_labels = torch.from_numpy(train_labels)
    validation_boards = torch.from_numpy(validation_boards)
    validation_labels = torch.from_numpy(validation_labels)

    train_loader = DataLoader(list(zip(train_boards, train_labels)), shuffle=True, batch_size=batch_size)
    validation_loader = DataLoader(list(zip(validation_boards, validation_labels)), shuffle=True, batch_size=batch_size)


    GoBot = load_model("model_test.pth", device)
    optim = torch.optim.Adam(GoBot.parameters(), learning_rate)

    training_loop(GoBot, device, train_loader, validation_loader, optim, loss_fn, num_epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
    args = parser.parse_args()
    main(args.num_epochs)
else:
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 0.0001
    batch_size = 1150

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using ", device)

    boards, labels = load_data()
    train_boards, train_labels, validation_boards, validation_labels = prepare_data(boards, labels)

    train_boards = torch.from_numpy(train_boards)
    train_labels = torch.from_numpy(train_labels)
    validation_boards = torch.from_numpy(validation_boards)
    validation_labels = torch.from_numpy(validation_labels)

    train_loader = DataLoader(list(zip(train_boards, train_labels)), shuffle=True, batch_size=batch_size)
    validation_loader = DataLoader(list(zip(validation_boards, validation_labels)), shuffle=True, batch_size=batch_size)
    
