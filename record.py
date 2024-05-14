import argparse
import pickle
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from preprocessing import load_np_data, filter_eeg_data, compute_psd, compute_bands
from dataset import EEGDataset
from mlp import MLP, finetune_model
from sklearn.svm import SVC
import pyautogui

STREAM_NAME = 'OpenBCI_EEG'

def init_stream():
    params = BrainFlowInputParams()
    params.serial_port = "/dev/cu.usbserial-DM00CVFL"   # change this to match the actual serial port, if changed

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
    # psd = compute_psd(filtered_data)
    # delta, theta, alpha, beta, gamma = compute_bands(psd)

    # make numpy array with the bands and labels
    # labeled_data = np.array([delta, theta, alpha, beta, gamma, label])
    # add labels as last column to filtered_data
    unlabeled_data = filtered_data._data.transpose()
    unlabeled_data = (unlabeled_data - unlabeled_data.mean(axis=0)) / unlabeled_data.std(axis=0)

    labels = np.full((unlabeled_data.shape[0], 1), label, dtype=int)

    return unlabeled_data, labels

def read_unlabeled_data(board):
    time.sleep(0.5)
    # data = board.get_current_board_data (256) # get latest 256 packages or less, doesnt remove them from internal buffer
    data = board.get_board_data()  # get all data and remove it from internal buffer

    raw = load_np_data(data)
    filtered_data = filter_eeg_data(raw)
    # psd = compute_psd(filtered_data)
    # delta, theta, alpha, beta, gamma = compute_bands(psd)
    
    # unlabeled_data = np.array([delta, theta, alpha, beta, gamma])
    unlabeled_data = filtered_data._data.transpose()
    unlabeled_data = (unlabeled_data - unlabeled_data.mean(axis=0)) / unlabeled_data.std(axis=0)

    return unlabeled_data

def stop_stream(board):
    board.stop_stream()
    board.release_session()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", help="calibrate input for eeg or eog", default=None)
    args = parser.parse_args()

    board = init_stream()

    if args.calibrate == 'eeg':
        print("Calibrating input...")

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load in pre-trained MLP model
        num_inputs = 4
        num_hidden = 500
        num_outputs = 2
        eeg_model = MLP(num_inputs=num_inputs, num_hidden=num_hidden, num_outputs=num_outputs)
        eeg_model.load_state_dict(torch.load('eeg_model_new.pt'))

        # Set require grad to false
        for param in eeg_model.parameters():
            param.requires_grad = False

        # Initialize new final layer of model
        eeg_model.fc = nn.Linear(num_hidden, num_outputs)

        # Define loss function and optimizer
        loss_fn = torch.nn.functional.nll_loss
        optimizer = torch.optim.Adam(eeg_model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=1, verbose=False)

        # Initialize calibration all_data
        all_data = np.array([])

        # Record the data
        for i in range(5):
            input("Press enter to record low focus...")
            data, labels = read_labeled_data(board, label=0)
            data = data[:, :-4]
            print(data)
            
            if all_data.size == 0:
                all_data = data
                all_labels = labels
            else:
                all_data = np.vstack((all_data, data), dtype=float)
                all_labels = np.vstack((all_labels, labels), dtype=int)
        
        print("Done recording low focus!\n")

        for i in range(5):
            input("Press enter to record high focus...")
            data, labels = read_labeled_data(board, label=1)
            data = data[:, :-4]
            print(data)

            all_data = np.vstack((all_data, data), dtype=float)
            all_labels = np.vstack((all_labels, labels), dtype=int)

        print("Done recording high focus!\n")

        # Calibrate the model
        print("Calibrating model...")

        # Create dataloader
        df = pd.DataFrame(all_data)
        df['focus'] = all_labels
        dataset = EEGDataset(df, -1)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        # fine tune model
        num_epochs = 5
        eeg_model = finetune_model(eeg_model, device, optimizer, scheduler, loss_fn, num_epochs, train_loader)

        # save the finetuned model
        torch.save(eeg_model.state_dict(), 'eeg_model_new.pt')

        print("Calibration complete! Run the script again to re-calibrate.")

    elif args.calibrate == 'eog':
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load in pre-trained MLP model
        num_inputs = 4
        num_hidden = 500
        num_outputs = 4
        eog_model = MLP(num_inputs=num_inputs, num_hidden=num_hidden, num_outputs=num_outputs)
        eog_model.load_state_dict(torch.load('eog_model_new.pt'))

        # Set require grad to false
        for param in eeg_model.parameters():
            param.requires_grad = False

        # Initialize new final layer of model
        eog_model.fc = nn.Linear(num_hidden, num_outputs)

        # Define loss function and optimizer
        loss_fn = torch.nn.functional.nll_loss
        optimizer = torch.optim.Adam(eog_model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=1, verbose=False)

        # Initialize calibration all_data
        all_data = np.array([])

        # Record the data
        for i in range(5):
            input("Press enter to record looking left...")
            data, labels = read_labeled_data(board, label=0)
            data = data[:, 4:]
            print(data)

            if all_data.size == 0:
                all_data = data
                all_labels = labels
            else:
                all_data = np.vstack((all_data, data), dtype=float)
                all_labels = np.vstack((all_labels, labels), dtype=int)

        print("Done recording looking left!\n")

        for i in range(5):
            input("Press enter to record looking right...")
            data, labels = read_labeled_data(board, label=1)
            data = data[:, 4:]
            print(data)

            all_data = np.vstack((all_data, data), dtype=float)
            all_labels = np.vstack((all_labels, labels), dtype=int)

        print("Done recording looking right!\n")

        for i in range(5):
            input("Press enter to record looking up...")
            data, labels = read_labeled_data(board, label=2)
            data = data[:, 4:]
            print(data)

            all_data = np.vstack((all_data, data), dtype=float)
            all_labels = np.vstack((all_labels, labels), dtype=int)

        print("Done recording looking up!\n")

        for i in range(5):
            input("Press enter to record looking down...")
            data, labels = read_labeled_data(board, label=3)
            data = data[:, 4:]
            print(data)

            all_data = np.vstack((all_data, data), dtype=float)
            all_labels = np.vstack((all_labels, labels), dtype=int)

        print("Done recording looking down!\n")

        # Calibrate the model
        print("Calibrating model...")
        # Create dataloader
        df = pd.DataFrame(all_data)
        df['direction'] = all_labels
        dataset = EEGDataset(df, -1)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

        # fine tune model
        num_epochs = 5
        eog_model = finetune_model(eog_model, device, optimizer, scheduler, loss_fn, num_epochs, train_loader)

        # save the finetuned model
        torch.save(eog_model.state_dict(), 'eog_model_new.pt')

        print("Calibration complete! Run the script again to re-calibrate.")

    else:
        print("Reading input...")

        # Load in pre-trained eeg model
        eeg_model = MLP(num_inputs=4, num_hidden=500, num_outputs=2)
        eeg_model.load_state_dict(torch.load('eeg_model_new.pt'))
        eeg_model.eval()

        # Load in pre-trained eog model
        eog_model = MLP(num_inputs=4, num_hidden=500, num_outputs=4)
        eog_model.load_state_dict(torch.load('eog_model_new.pt'))
        eog_model.eval()

        while True:
            print("Press Ctrl+C to stop.")
            # check keyboard interrupt
            try:
                # read data
                data = read_unlabeled_data(board)
                eeg_data = data[:, :-4]
                eog_data = data[:, 4:]
                # print("eeg:", eeg_data)
                # print("eog:", eog_data)

                # predict eeg
                data = torch.from_numpy(eeg_data).float()
                output = eeg_model(data)
                # apply softmax
                pred = torch.nn.functional.softmax(output, dim=1)
                # get max probability prediction
                pred = torch.argmax(pred, dim=1)
                # p_prev = None
                for p in pred:
                    if p == 0:
                        print("Low focus.")
                    elif p == 1:
                        print("High focus.")
                        pyautogui.press('space')
                    # p_prev = p

                # predict eog
                data = torch.from_numpy(eog_data).float()
                output = eog_model(data)
                # apply softmax
                pred = torch.nn.functional.softmax(output, dim=1)
                # get max probability prediction
                pred = torch.argmax(pred, dim=1)
                # p_prev = None
                for p in pred:
                    if np.max(p) < 0.75:
                        print("No prediction.")
                        # p_prev = -1
                    elif p.argmax() == 0:
                        print("Looking left.")
                        pyautogui.press('a')
                        # p_prev = 0
                    elif p.argmax() == 1:
                        print("Looking right.")
                        pyautogui.press('d')
                        # p_prev = 1
                    elif p.argmax() == 2:
                        print("Looking up.")
                        pyautogui.press('w')
                        # p_prev = 2
                    elif p.argmax() == 3:
                        print("Looking down.")
                        pyautogui.press('s')
                        # p_prev = 3

            except KeyboardInterrupt:
                break

    stop_stream(board)
    print("Stream closed.")

if __name__ == "__main__":
    main()