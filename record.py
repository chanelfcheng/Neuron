import argparse
import time
import numpy as np
import pandas as pd
import torch
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from preprocessing import load_np_data, filter_eeg_data, compute_psd, compute_bands
from snn import SNN

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

        # Initialize model with device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SNN(device)

        # Load in pre-trained model parameters
        model.load_state_dict(torch.load('snn.pt'))

        # Set require grad to false
        for param in model.parameters():
            param.requires_grad = False

        # Initialize new final layer of model
        model.fc = torch.nn.Linear(5, 1)

        # Initialize calibration dataset
        dataset = np.array([])

        for i in range(3):
            input("Press enter to calibrate high focus...")
            data = read_labeled_data(board, label=1)
            print(data)

            if dataset.size == 0:
                dataset = data
            else:
                dataset = np.vstack((dataset, data))

        print("Done calibrating high focus!\n")
        
        for i in range(3):
            input("Press enter to calibrate low focus...")
            data = read_labeled_data(board, label=0)
            print(data)
            
            dataset = np.vstack((dataset, data))
        
        print("Done calibrating low focus!\n")

        print("Run the script again to re-calibrate.")

        
    else:
        print("Reading input...")

        while True:
            print("Press Ctrl+C to stop.")
            # check keyboard interrupt
            try:
                # read data
                data = read_unlabeled_data(board)
                print(data)
            except KeyboardInterrupt:
                break

    stop_stream(board)
    print("Stream closed.")

if __name__ == "__main__":
    main()