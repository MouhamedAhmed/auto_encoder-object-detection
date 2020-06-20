import numpy as np
from datetime import datetime 

import torch
import torch.nn as nn
import torch.nn.functional as F


import matplotlib.pyplot as plt

from model import *
from train_val import *

# check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# parameters
RANDOM_SEED = 42
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
N_EPOCHS = 1

IMG_SIZE = 256


# instantiate the model
torch.manual_seed(RANDOM_SEED)

model = Autoencoder().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
cross_entropy_loss_criterion = nn.BCELoss()

# start training
model, optimizer, train_losses, valid_losses = training_loop(model, cross_entropy_loss_criterion,BATCH_SIZE, optimizer, N_EPOCHS, DEVICE)
 

