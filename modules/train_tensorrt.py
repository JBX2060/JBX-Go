import sys
import torch
import tensorrt as trt
from modules.early_stopping import EarlyStopper
from modules.visualize import create_go_board_image

from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1 << 20
        with open(onnx_file_path, 'rb') as model:
            parser.parse(model.read())
        return builder.build_cuda_engine(network)

def torch2trt(model, inputs):
    onnx_file_path = 'model.onnx'
    torch.onnx.export(model, inputs, onnx_file_path, opset_version=11, verbose=True)
    engine = build_engine(onnx_file_path)
    return engine

def trt_inference(engine, inputs):
    inputs_trt = inputs.detach().cpu().numpy()
    # Allocate buffers
    h_input = inputs_trt.ravel()
    h_output = np.empty(engine.get_binding_shape(1), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()

    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(d_input, h_input, stream)
    # Run inference.
    with engine.create_execution_context() as context:
        context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    # Synchronize the stream
    stream.synchronize()

    return torch.tensor(h_output.reshape(inputs.shape[0], -1))


def training_loop(model, device, train_loader, test_loader, optim, loss_fn, n_epochs=10000, patience=4):
    scaler = GradScaler()

    for epoch_idx in range(n_epochs):
        correct = 0
        wrong = 0
        total_loss = 0
        test_total_loss = 0
        test_correct = 0
        test_wrong = 0

        print(f"EPOCH {epoch_idx}")

        for inputs, labels in tqdm(train_loader, desc="Training"):
            board_batch, labels_batch = inputs.to(device), labels.to(device)

            with autocast():
                output = model(board_batch)
                try:
                    loss = loss_fn(output, labels_batch)
                except:
                    print("error occured invalid training data")
                    continue

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()

            output_index = output.argmax(dim=1)
            correct += (output_index == labels_batch).sum().item()
            wrong += (output_index != labels_batch).sum().item()

            total_loss += loss.item()

        # Convert the PyTorch model to a TensorRT engine
        inputs, _ = next(iter(test_loader))
        engine = torch2trt(model, inputs.to(device))

        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            board_batch, labels_batch = inputs.to(device), labels.to(device)

            with torch.no_grad():
                validation_output = trt_inference(engine, board_batch)
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