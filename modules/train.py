import sys
import torch
from modules.visualize import create_go_board_image

from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

def training_loop(model, device, train_loader, validation_loader, optim, loss_fn, n_epochs=10000, patience=4):

    scaler = GradScaler()
    best_val_loss = float('inf')
    no_improvement_counter = 0

    for epoch_idx in range(n_epochs):
        correct = 0
        wrong = 0
        total_loss = 0
        test_total_loss = 0
        test_correct = 0
        test_wrong = 0

        print(f"EPOCH {epoch_idx}")

        # Wrap the train_loader with tqdm for a progress bar
        for inputs, labels in tqdm(train_loader, desc="Training"):
            board_batch, labels_batch = inputs.to(device), labels.to(device)

            # Use autocast for mixed-precision training
            with autocast():
                output = model(board_batch)
                try:
                    loss = loss_fn(output, labels_batch)
                except:
                    print("error occured invalid training data")
                    continue

            # Optimize the gradients using GradScaler
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()

            output_index = output.argmax(dim=1)
            correct += (output_index == labels_batch).sum().item()
            wrong += (output_index != labels_batch).sum().item()

            total_loss += loss.item()

            # create_go_board_image(board_batch[0].cpu().numpy(), f"images/epoch_{epoch_idx}_batch_{i}.png")


        # Wrap the validation_loader with tqdm for a progress bar
        for inputs, labels in tqdm(validation_loader, desc="Evaluating"):

            board_batch, labels_batch = inputs.to(device), labels.to(device)
            # print(board_batch.shape, labels_batch.shape)
            
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

        validation_loss = test_total_loss / total_test_times
        loss = total_loss / total_times

        print("Accuracy: ", accuracy)
        print("Test Accuracy: ", test_accuracy)

        print(f"Avg Loss: {loss}")
        print(f"Avg Test Loss: {validation_loss}")

        # Update best_val_loss and no_improvement_counter
        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1

        # Check for early stopping
        if no_improvement_counter >= patience:
            print("Early stopping triggered")
            print("We are at epoch:", epoch_idx)
            break

        try:
            torch.save(model.state_dict(), "model.pth")

        except KeyboardInterrupt:
            print("Saving ...")
            torch.save(model.state_dict(), "model.pth")
            sys.exit()
