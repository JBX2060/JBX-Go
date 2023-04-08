import torch
from torch import nn
from data_preparation import load_data, prepare_data
from model import Model, load_model
from train import training_loop
from torch.utils.data import DataLoader

def main():
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

    train_boards = train_boards.to(device)
    train_labels = train_labels.to(device)
    validation_boards = validation_boards.to(device)
    validation_labels = validation_labels.to(device)

    train_loader = DataLoader(list(zip(train_boards, train_labels)), shuffle=True, batch_size=batch_size)
    validation_loader = DataLoader(list(zip(validation_boards, validation_labels)), shuffle=True, batch_size=batch_size)


    GoBot = load_model("model_test.pth", device)
    optim = torch.optim.Adam(GoBot.parameters(), learning_rate)

    trained_model = training_loop(GoBot, device, train_loader, validation_loader, optim, loss_fn, n_epochs=10000)
    torch.save(trained_model.state_dict(), "model_test.pth")

if __name__ == "__main__":
    main()