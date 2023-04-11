import os
import numpy as np
from sklearn.model_selection import train_test_split
from modules.convert_sgf import process_files_in_parallel


# 'boards_extra_full', 'labels_extra_full', 'boards_full.npy', 'labels_full.npy', 'boards_large.npy', 'labels_large.npy', 'boards.npy', 'labels.npy'
processed_data_path = 'bot_data/kata2'

boards_path = '/home/jbx2060/JBX2020/proccesed_data/kata2_boards.npy'
labels_path = '/home/jbx2060/JBX2020/proccesed_data/kata2_labels.npy'



def load_data(boards_path=boards_path, labels_path=labels_path, processed_data_path=processed_data_path):
    if os.path.exists(boards_path) and os.path.exists(labels_path):
        print("Loading Training Dataset...")
        boards = np.load(boards_path)
        labels = np.load(labels_path
    )
     
    else:
        print("Processing data...")
        file_paths = [os.path.join(processed_data_path, file_name) for file_name in os.listdir(processed_data_path)]
        boards, labels = process_files_in_parallel(file_paths)

        np.save(boards_path, boards)
        np.save(labels_path, labels)

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
