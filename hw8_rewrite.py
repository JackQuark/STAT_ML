# b12209017
# reference: code by Yi-Jhen Zeng at NTU, May 2022
# ==================================================
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from time import perf_counter

device = torch.device("cpu")
# ==================================================

def data_split(xdata, ydata, test_size=0.3):    
    n_samples = xdata.shape[0]
    n_train = int(n_samples * (1 - test_size))

    x_train, x_test = np.split(xdata, [n_train])
    y_train, y_test = np.split(ydata, [n_train])
    return x_train, x_test, y_train, y_test

# ==================================================

class LogisticRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        return torch.sigmoid(self.linear(x))
    
    def train(self, x, y, optimizer: optim.Optimizer):
        optimizer.zero_grad()
        y_pred = torch.squeeze(self(x))
        loss = F.binary_cross_entropy(y_pred, y)
        loss.backward()
        optimizer.step() # update parameters
        acc = (y_pred.round() == y).float().mean()
        return loss.item(), acc.item()
    
    def test(self, x, y):
        y_pred = torch.squeeze(self(x))
        loss = F.binary_cross_entropy(y_pred, y)
        acc = (y_pred.round() == y).float().mean()
        return loss.item(), acc.item()

# ==================================================
def main():
    xdata = np.loadtxt("./hw8_u_x.csv", delimiter=",", skiprows=1, usecols=range(1, 145)) # samples(200)*lon(144)
    ydata = np.loadtxt("./hw8_u_y.csv", delimiter=",", skiprows=1, usecols=[1]) # samples(200)
    x_train, x_test, y_train, y_test = data_split(xdata, ydata, test_size=0.3)
    # to torch tensor (float32)
    x_train = torch.from_numpy(x_train).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    x_test = torch.from_numpy(x_test).float().to(device)
    y_test = torch.from_numpy(y_test).float().to(device)
    
    # ==============================
    # hyperparameters
    input_size  = 144
    output_size = 1
    EPOCHS      = 100
    lr          = 0.001  # learning rate
    wd          = 0.1    # weight decay
    # ==============================
    model = LogisticRegression(144, 1)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd) # stochastic gradient descent
    
    train_rec = np.zeros((2, EPOCHS)) # record train loss and acc
    test_rec = np.zeros((2, EPOCHS))
    for epoch in range(EPOCHS):
        train_rec[0, epoch], train_rec[1, epoch] = model.train(x_train, y_train, optimizer)
        test_rec[0, epoch], test_rec[1, epoch] = model.test(x_test, y_test)
        
        print(f"Epoch [{epoch+1:3d}/{EPOCHS}]")
        print(f"train loss: {train_rec[0, epoch]:.4f}, acc: {train_rec[1, epoch]:.4f}")
        print(f"test  loss: {test_rec[0, epoch]:.4f}, acc: {test_rec[1, epoch]:.4f}")
        
    fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax[0].plot(range(EPOCHS), train_rec[0], label='train')
    ax[1].plot(range(EPOCHS), train_rec[1], label='train')
    ax[0].plot(range(EPOCHS), test_rec[0], label='test')
    ax[1].plot(range(EPOCHS), test_rec[1], label='test')
    
    ax[0].set_title('Loss')
    ax[1].set_title('Accuracy')
    ax[1].set_xlabel('Epoch')
    
# ==================================================
if __name__ == '__main__':
    start_t = perf_counter()
    main()
    end_t = perf_counter()
    print(f"time: {(end_t - start_t)*1000:.3f} ms")
# ==================================================