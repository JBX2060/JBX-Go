import sys
import torch
from modules.early_stopping import EarlyStopper
from modules.visualize import create_go_board_image

from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

def training_loop(model, device, train_loader, test_loader, optim, loss_fn, n_epochs=10000):
    scaler = GradScaler()

    for epoch_idx in range(n_epochs):
        correct = 0
        wrong = 0
        total_loss = 0
        test_total_loss = 0
        test_correct = 0
        test_wrong = 0

        print(f"EPOCH {epoch_idx}")

        # Wrap the train_loader with tqdm for a progress bar
        for i, (inputs, labels) in tqdm(enumerate(train_loader), desc="Training"):
            board_batch, labels_batch = inputs.to(device), labels.to(device)

            # Use autocast for mixed-precision training
            with autocast():
                output = model(board_batch)
                loss = loss_fn(output, labels_batch)

            # Optimize the gradients using GradScaler
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()

            output_index = output.argmax(dim=1)
            correct += (output_index == labels_batch).sum().item()
            wrong += (output_index != labels_batch).sum().item()

            total_loss += loss.item()

            create_go_board_image(board_batch[0].cpu().numpy(), f"images/epoch_{epoch_idx}_batch_{i}.png")


        # Wrap the test_loader with tqdm for a progress bar
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            board_batch, labels_batch = inputs.to(device), labels.to(device)

            # Validation
            with torch.no_grad(), autocast():
                validation_output = model(board_batch)
                test_loss = loss_fn(validation_output, labels_batch)

            validation_output = validation_output.argmax(dim=1)
            test_correct += (validation_output == labels_batch).sum().item()
            test_wrong += (validation_output != labels_batch).sum().item()

            test_total_loss += test_loss.item()

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
            torch.save(model.state_dict(), "model_test.pth")

        except KeyboardInterrupt:
            print("Saving ...")
            torch.save(model.state_dict(), "model_test.pth")
            sys.exit()
