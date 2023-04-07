import os
import numpy as np
from sklearn.model_selection import train_test_split
from convert_sgf import process_files_in_parallel


def load_data():
    if os.path.exists("boards.npy") and os.path.exists("labels.npy"):
        print("Loading Training Dataset...")
        boards = np.load("boards.npy")
        labels = np.load("labels.npy")
    else:
        print("Processing data...")
        file_paths = [os.path.join("go_data/9d", file_name) for file_name in os.listdir("go_data/9d")]
        boards, labels = process_files_in_parallel(file_paths)

        np.save("boards.npy", boards)
        np.save("labels.npy", labels)

    return boards, labels


def prepare_data(boards, labels):
    data = list(zip(boards, labels))
    train_data, validation_data = train_test_split(data, test_size=0.2, random_state=42)

    train_boards, train_labels = zip(*train_data)
    validation_boards, validation_labels = zip(*validation_data)

    train_boards = np.stack(train_boards)
    train_labels = np.stack(train_labels)
    validation_boards = np.stack(validation_boards)
    validation_labels = np.stack(validation_labels)

    return train_boards, train_labels, validation_boards, validation_labels
