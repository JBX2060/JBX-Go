import os
import torch
import numpy as np
import sys

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from modules.convert_sgf import process_files_in_parallel
from sklearn.model_selection import train_test_split
from modules.visualize import create_go_board_image
from modules.model import Model
from main import validation_loader, boards, loss_fn, device

from torch.cuda.amp import GradScaler, autocast

from tqdm import tqdm

def adjust_channels(param, numx_channels, interpolation_mode='bilinear'):
    input_channels = param.size(0)
    source_weight = param.unsqueeze(0)

    print(f"Input channels {input_channels}, numx channels {numx_channels}")
    print(f"source weight {source_weight.shape}, size: {(1, numx_channels, *source_weight.shape[2:])}")
    
    if input_channels != numx_channels:
        print(f"Adjusting channels from {input_channels} to {numx_channels}")
        adjusted_weight = torch.nn.functional.interpolate(source_weight, 
                                                           size=(1, numx_channels, *source_weight.shape[2:]),
                                                           mode=interpolation_mode,
                                                           align_corners=False)
    else:
        adjusted_weight = source_weight

    return adjusted_weight.squeeze(0)

# Helper function to upscale or downscale the model weights.
def fit_pretrained_weights(source_state_dict, target_model):
    target_state_dict = target_model.state_dict()
    for name, param in source_state_dict.items():
        if name not in target_state_dict:
            continue
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data

        # print(f"param {param.shape}, target {target_state_dict[name].shape}")

        adjusted_param = adjust_channels(param, target_state_dict[name].size(0))

        print(f"Adjusted param, {adjusted_param.size()}, target {target_state_dict[name].size()}")

        if adjusted_param.size() != target_state_dict[name].size():
            # Scale the weights
            target_state_dict[name].copy_(torch.nn.functional.interpolate(adjusted_param.unsqueeze(0),
                                                                          size=target_state_dict[name].size()[2:],
                                                                          mode='bilinear',
                                                                          align_corners=False).squeeze(0))
        else:
            target_state_dict[name].copy(adjusted_param)
    return target_model

import random
#create new_model #Muation functions.
def create_new_model(model):
    numx_res_blocks = random.randint(1, 100)
    numx_channels = random.randint(32, 512)  # You can adjust the range as needed
    # Transfer the weights of the pre-trained model
    state_dict_x = model.state_dict()
    child_model =  Model(num_res_blocks=numx_res_blocks, channels=numx_channels)
    child_model_loaded = fit_pretrained_weights(state_dict_x, child_model)

    child_model_loaded.to(device)

    return child_model_loaded

# Create a fitness function
def fitness(model):
    train_loss = 0.0
    num_correct = 0
    num_samples = 0
    scaler = GradScaler()
    with torch.no_grad():
        for inputs, targets in validation_loader:
            # Move the inputs and targets to the device (e.g. GPU)
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass with autocast
            with autocast():
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

            # Update the test loss and accuracy
            train_loss += loss.item() * inputs.size(0)
            _, predictions = torch.max(outputs, 1)
            num_correct += (predictions == targets).sum().item()
            num_samples += targets.size(0)

    train_loss /= len(boards)

    return train_loss


# Logic for the greedy or evolutionary
population  = 20
generations = 10

global_model_num = 0
models = {} # Index 0: Model intance, Index 1: fitness score

model = Model()

if torch.cuda.is_available():
    # Move the model parameters to the GPU
    model.cuda()

model.load_state_dict(torch.load("model_test.pth"))
print("Intialized the base model!")
models[f'Model_{global_model_num}'] = [] # Instance
print("Appending the model")
models[f'Model_{global_model_num}'].extend([model, fitness(model)]) # Tuple of (instance, fitness)
global_model_num+= 1

for i in tqdm(range(generations), desc="Generations"):
    if i != 0:
        #Sort the models based on their fitness scores
        ranked_models = dict(sorted(models.items(), key=lambda x: x[1][1]))
        # Pick the best model
        for i, k in enumerate(list(models.keys())):
            if i != 0:
                del models[k]

    for j in range(population-1):
        child_model = create_new_model(next(iter(models.values()))[0])
        print("CUDA:", next(child_model.parameters()).is_cuda) # returns a boolean
        models[f'Model_{global_model_num}'] = []
        models[f'Model_{global_model_num}'].extend([child_model, fitness(child_model)])
    
    print(models)
    print(f"Generation {i}")