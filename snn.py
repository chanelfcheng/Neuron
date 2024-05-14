from typing import NamedTuple
from tqdm import tqdm, trange
import numpy as np
import torch
import torch.nn as nn
from norse.torch import (
    SpikeLatencyLIFEncoder,
    LIFParameters,
    LIFState,
    LICell,
    LIState
)
from norse.torch.functional import (
    lif_step,
    lift,
    lif_feed_forward_step,
    lif_current_encoder,
    LIFParameters
)
from norse.torch.module.lif import LIFCell, LIFRecurrentCell


class SNNState(NamedTuple):
    lif0: LIFState
    readout: LIState

class SNN(nn.Module):
    def __init__(
        self, num_inputs=4, num_hidden=100, num_outputs=2, record=False, dt=0.001
    ):
        super(SNN, self).__init__()
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.record = record
        self.dt = dt

        self.layer1 = LIFRecurrentCell(
            num_inputs,
            num_hidden,
            p=LIFParameters(alpha=torch.as_tensor(0.0001), v_th=torch.as_tensor(0.5), v_leak=torch.as_tensor(0.25), v_reset=torch.as_tensor(0.0)),
            dt=dt
        )

        self.act = nn.ReLU()
        self.output_layer = nn.Linear(num_hidden, num_outputs)
        self.output_leaky = LICell(dt=dt)

    def forward(self, x):
        seq_length, batch_size = x.shape[0], x.shape[1]
        s1 = so = None
        voltages = []

        if self.record:
            self.recording = SNNState(
                LIFState(
                    z=torch.zeros(seq_length, batch_size, self.num_hidden),
                    v=torch.zeros(seq_length, batch_size, self.num_hidden),
                    i=torch.zeros(seq_length, batch_size, self.num_hidden),
                ),
                LIState(
                    v=torch.zeros(seq_length, batch_size, self.num_outputs),
                    i=torch.zeros(seq_length, batch_size, self.num_outputs),
                ),
            )

        for t in range(seq_length):
            z = x[t, :, :].view(-1, self.num_inputs)
            z, s1 = self.layer1(z, s1)
            z = self.act(z)
            z = self.output_layer(z)
            vo, so = self.output_leaky(z, so)

            if self.record:
                self.recording.lif0.z[t, :, :] = s1.z
                self.recording.lif0.v[t, :, :] = s1.v
                self.recording.lif0.i[t, :, :] = s1.i
                self.recording.readout.v[t, :] = so.v
                self.recording.readout.i[t, :] = so.i
              
            voltages.append(vo)

        return torch.stack(voltages)


class SpikingModel(nn.Module):
    def __init__(self, snn, encoder, decode_last=False):
        super(SpikingModel, self).__init__()
        self.snn = snn
        self.encoder = encoder
        self.decode_last = decode_last

    def forward(self, x):
        x = self.encoder(x)
        x = self.snn(x)
        log_p_y = self.decode_spikes(x) if not self.decode_last \
            else self.decode_last_spike(x)
        return log_p_y
    
    def decode_spikes(self, x):
        x, _ = torch.max(x, 0)
        log_p_y = nn.functional.log_softmax(x, dim=1)
        return log_p_y

    def decode_last_spike(self, x):
        x = x[-1]
        log_p_y = nn.functional.log_softmax(x, dim=1)
        return log_p_y
    
    
def train_model(model, device, optimizer, scheduler, loss_fn, num_epochs, train_loader, val_loader):
    # send model to device
    model.to(device)

    # set model to training mode
    model.train()

    # initialize losses
    losses = []
    train_losses = []
    val_losses = []

    # train model
    for epoch in range(num_epochs):
        for (data, target) in tqdm(train_loader, leave=False):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step(metrics=loss)

        mean_loss = np.mean(losses)
        train_losses.append(mean_loss)
        print(f"Train epoch {epoch+1}/{num_epochs} loss: {mean_loss:.4f}")

        # evaluate model
        val_loss, _ = test_model(model, device, loss_fn, val_loader)
        val_losses.append(val_loss)

    return model, train_losses, val_losses


def test_model(model, device, loss_fn, test_loader):
    # send model to device
    model.to(device)

    # set model to evaluation mode
    model.eval()

    # initialize test loss and number of correct predictions
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for (data, target) in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(
                output, target
            ).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100.0 * correct / len(test_loader.dataset)

        print(f"Test loss: {test_loss:.4f}")
        print(f"Validation accuracy: {accuracy:.2f}%")
        
    return test_loss, accuracy


def finetune_model(model, device, optimizer, loss_fn, num_epochs, train_loader):
    # send model to device
    model.to(device)

    # set model to training mode
    model.train()

    # initialize losses
    losses = []

    # train model
    for epoch in range(num_epochs):
        for (data, target) in tqdm(train_loader, leave=False):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        mean_loss = np.mean(losses)
        print(f"Train epoch {epoch+1}/{num_epochs} loss: {mean_loss:.4f}")

    # save model
    torch.save(model.state_dict(), "snn_finetune.pt")