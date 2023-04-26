import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import snntorch
import snntorch.functional as SF
from snntorch import surrogate
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from preprocessing import load_np_data, filter_eeg_data, compute_psd, compute_bands
from dataset import EEGDataset
from snn import SNN, finetune_snn

STREAM_NAME = 'OpenBCI_EEG'

def init_stream():
    params = BrainFlowInputParams()
    params.serial_port = "/dev/cu.usbserial-DM00CVFL"

    board = BoardShim(BoardIds.CYTON_BOARD, params)
    board.prepare_session()
    board.start_stream()

    return board

def read_labeled_data(board, label):
    time.sleep(0.5)
    # data = board.get_current_board_data (256) # get latest 256 packages or less, doesnt remove them from internal buffer
    data = board.get_board_data()  # get all data and remove it from internal buffer

    raw = load_np_data(data)
    filtered_data = filter_eeg_data(raw)
    psd = compute_psd(filtered_data)
    delta, theta, alpha, beta, gamma = compute_bands(psd)

    # make numpy array with the bands and labels
    data = np.array([delta, theta, alpha, beta, gamma, label])

    return data

def read_unlabeled_data(board):
    time.sleep(0.5)
    # data = board.get_current_board_data (256) # get latest 256 packages or less, doesnt remove them from internal buffer
    data = board.get_board_data()  # get all data and remove it from internal buffer

    raw = load_np_data(data)
    filtered_data = filter_eeg_data(raw)
    psd = compute_psd(filtered_data)
    delta, theta, alpha, beta, gamma = compute_bands(psd)
    
    data = np.array([delta, theta, alpha, beta, gamma])

    return data

def stop_stream(board):
    board.stop_stream()
    board.release_session()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", help="whether to calibrate or not", action="store_true")
    args = parser.parse_args()

    board = init_stream()

    if args.calibrate:
        print("Calibrating input...")

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load in pre-traind model
        model = SNN()
        model.load_state_dict(torch.load('snn.pt'))

        # Set require grad to false
        for param in model.parameters():
            param.requires_grad = False

        # Initialize new final layer of model
        num_hidden = 100
        pop_outputs = 100
        beta = 0.9
        grad = surrogate.fast_sigmoid()
        model.output_layer = nn.Linear(num_hidden, pop_outputs)
        model.output_leaky = snntorch.Leaky(beta=beta, spike_grad=grad, init_hidden=True, output=True)

        # Define loss function and optimizer
        loss_fn = SF.mse_count_loss(correct_rate=1.0, incorrect_rate=0.0, population_code=True, num_classes=2)
        optimizer = torch.optim.Adam(snntorch.net.parameters(), lr=1e-3, betas=(0.9, 0.999))

        # Initialize calibration dataset
        dataset = np.array([])

        # Record the data
        for i in range(3):
            input("Press enter to record high focus...")
            data = read_labeled_data(board, label=1)
            print(data)

            if dataset.size == 0:
                dataset = data
            else:
                dataset = np.vstack((dataset, data))

        print("Done recording high focus!\n")
        
        for i in range(3):
            input("Press enter to record low focus...")
            data = read_labeled_data(board, label=0)
            print(data)
            
            dataset = np.vstack((dataset, data))
        
        print("Done recording low focus!\n")

        # Calibrate the model
        print("Calibrating model...")

        # Create dataloader
        torch_data = torch.from_numpy(dataset).float()
        dataset = EEGDataset(torch_data, -1)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

        # fine tune model
        num_epochs = 1
        finetune_snn(model, device, optimizer, loss_fn, num_epochs, train_loader)

        print("Calibration complete! Run the script again to re-calibrate.")

        
    else:
        print("Reading input...")

        # Load in pre-traind model
        model = SNN()
        model.load_state_dict(torch.load('snn.pt'))
        model.eval()

        while True:
            print("Press Ctrl+C to stop.")
            # check keyboard interrupt
            try:
                # read data
                data = read_unlabeled_data(board)
                print(data)

                # predict
                torch_data = torch.from_numpy(data).float()
                spk_rec, _ = model(torch_data)
                print(spk_rec.shape)
            except KeyboardInterrupt:
                break

    stop_stream(board)
    print("Stream closed.")

if __name__ == "__main__":
    main()