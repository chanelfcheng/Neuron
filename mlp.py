import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

class MLP(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()

        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        self.layer1 = nn.Linear(num_inputs, num_hidden)
        # self.layer2 = nn.Linear(num_hidden, num_hidden*2)
        # self.layer3 = nn.Linear(200, 500)
        # self.layer4 = nn.Linear(500, 200)
        # self.layer5 = nn.Linear(num_hidden*2, num_hidden)
        self.fc = nn.Linear(num_hidden, num_outputs)

        self.act = nn.ReLU()

    def forward(self, x, return_embedding=False):
        embed = self.act(self.layer1(x))
        # x = self.act(self.layer2(x))
        # x = self.act(self.layer3(x))
        # x = self.act(self.layer4(x))
        # embed = self.act(self.layer5(x))
        out = self.fc(embed)
        # out = self.softmax(out)

        if return_embedding:
            return out, embed
        else:
            return out
        

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


def finetune_model(model, device, optimizer, scheduler, loss_fn, num_epochs, train_loader):
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
        scheduler.step(metrics=loss)

        mean_loss = np.mean(losses)
        print(f"Train epoch {epoch+1}/{num_epochs} loss: {mean_loss:.4f}")

    return model