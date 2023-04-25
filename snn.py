import torch
import snntorch
import torch.nn as nn
from snntorch import surrogate, utils, backprop
import snntorch.functional as SF

class SNN(nn.Module):
    def __init__(self, num_inputs=5, num_hidden=128, num_outputs=2, 
                 num_steps=2, alpha=0.9, beta=0.9, 
                 grad=surrogate.fast_sigmoid(), pop_outputs=100):
        super(SNN, self).__init__()
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.num_steps = num_steps
        self.beta = beta
        self.grad = grad
        self.pop_outputs = pop_outputs
        
        # self.net = nn.Sequential(nn.Flatten(),
        #             nn.Linear(num_inputs, num_hidden),
        #             snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True),
        #             nn.Linear(num_hidden, pop_outputs),
        #             snn.Leaky(beta=beta, spike_grad=grad, init_hidden=True, output=True)
        #             ).to(device)

        self.flatten = nn.Flatten()
        self.input_layer = nn.Linear(num_inputs, num_hidden)
        self.input_leaky = snntorch.Synaptic(alpha=alpha, beta=beta, spike_grad=grad, init_hidden=True)
        self.output_layer = nn.Linear(num_hidden, pop_outputs)
        self.output_leaky = snntorch.Leaky(beta=beta, spike_grad=grad, init_hidden=True, output=True)

    def forward(self, x):
        # return self.net(x)
        if x.dim() != 1:
          x = self.flatten(x)
        x = self.input_layer(x)
        x = self.input_leaky(x)
        x = self.output_layer(x)
        x = self.output_leaky(x)

        return x
    
def train_snn(snn, device, optimizer, loss_fn, num_epochs, train_loader, val_loader):
    # send model to device
    snn.to(device)
    
    # train network
    snn.train()

    for epoch in range(num_epochs):
        avg_loss = backprop.BPTT(snn, train_loader, num_steps=snn.num_steps,
                            optimizer=optimizer, criterion=loss_fn, time_var=False, device=device)

        print(f"Epoch: {epoch}")
        print(f"Validation set accuracy: {test_accuracy(snn, device, val_loader)*100:.3f}%\n")
    
    # save model
    torch.save(snn.state_dict(), "snn.pt")

    return avg_loss

def test_accuracy(snn, device, data_loader, population_code=True):
  # send model to device
  snn.to(device)

  with torch.no_grad():
    total = 0
    acc = 0
    snn.eval()

    data_loader = iter(data_loader)
    for data, targets in data_loader:
      data = data.to(device)
      targets = targets.to(device)
      utils.reset(snn)
      spk_rec, _ = snn(data)

      if population_code:
        acc += SF.accuracy_rate(spk_rec.unsqueeze(0), targets, population_code=True, num_classes=2) * spk_rec.size(1)
      else:
        acc += SF.accuracy_rate(spk_rec.unsqueeze(0), targets) * spk_rec.size(1)

      total += spk_rec.size(1)

  return acc/total

def finetune_snn(snn, device, optimizer, loss_fn, num_epochs, train_loader):
    # send model to device
    snn.to(device)
    
    # train network
    snn.train()

    for epoch in range(num_epochs):
        avg_loss = backprop.BPTT(snn, train_loader, num_steps=snn.num_steps,
                            optimizer=optimizer, criterion=loss_fn, time_var=False, device=device)

        print(f"Epoch: {epoch}")
    
    # save model
    torch.save(snn.state_dict(), "snn_finetune.pt")

    return avg_loss