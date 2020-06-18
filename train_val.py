import numpy as np
from datetime import datetime 

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data import DataLoader

from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from dataloader import *
import json


###############
# train iteration
def train(train_set, batch_size, model, cross_entropy_loss_criterion, optimizer, device):
    '''
    Function for the training step of the training loop
    '''

    model.train()
    epoch_loss = 0

    l = len(train_set)/batch_size
    i = 0

    while len(train_set)>0:
         # get batch
        batch = get_batch(train_set,batch_size)
        normalize_batch(batch)
        X, y_true = convert_batch_to_tensors(batch)
        X = X.to(device)
        y_true = y_true.to(device)
        

        optimizer.zero_grad()


        # Forward pass
        y_hat = model(X)
        
        # loss
        cross_entropy_loss = cross_entropy_loss_criterion(y_hat, y_true) 
        
        epoch_loss += cross_entropy_loss.item()
        
        i += 1        

        # Backward pass
        cross_entropy_loss.backward()
        optimizer.step()


    epoch_loss = epoch_loss / l
    
    return model,optimizer, epoch_loss


# validate 
def validate(test_set, batch_size, model, cross_entropy_loss_criterion, device):
    '''
    Function for the validation step of the training loop
    '''
   
    model.eval()
    epoch_loss = 0
    
    l = len(test_set)/batch_size

    while len(test_set)>0:
        # get batch
        batch = get_batch(test_set,batch_size)
        normalize_batch(batch)
        X, y_true = convert_batch_to_tensors(batch)

        X = X.to(device)
        y_true = y_true.to(device)
        
        # Forward pass
        y_hat = model(X) 
         
        # loss
        cross_entropy_loss = cross_entropy_loss_criterion(y_hat, y_true) 
      
        epoch_loss += cross_entropy_loss.item()

       
    epoch_loss = epoch_loss / l
        
    return model, epoch_loss


def training_loop(model, cross_entropy_loss_criterion, batch_size, optimizer, epochs, device, print_every=1):
    '''
    Function defining the entire training loop
    '''

    # set objects for storing metrics
    best_loss = 1e10
    train_losses = []
    valid_losses = []
    
    train_set_start, test_set_start,_ = load_data()
    # Train model
    for epoch in range(0, epochs):

        # load dataset
        train_set = copy.deepcopy(train_set_start)
        test_set = copy.deepcopy(test_set_start)

        # training
        model, optimizer, train_loss = train(train_set, batch_size, model, cross_entropy_loss_criterion, optimizer, device)
        train_losses.append(train_loss)
        # validation
        with torch.no_grad():
            model, valid_loss = validate(test_set, batch_size, model, cross_entropy_loss_criterion, device)
            valid_losses.append(valid_loss)

        if epoch % print_every == (print_every - 1):
            # # train_set, test_set = load_data()
            # train_set = copy.deepcopy(train_set_start)
            # test_set = copy.deepcopy(test_set_start)
            # train_acc = get_accuracy(model, train_set,batch_size, device=device)
            # valid_acc = get_accuracy(model, test_set,batch_size, device=device)
                
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  )

    plot_losses(train_losses, valid_losses)
    
    return model, optimizer, train_losses, valid_losses



############
# helper functions
def get_accuracy(model, dataset,batch_size, device):
    '''
    Function for computing the accuracy of the predictions over the entire data_loader
    '''
    
    correct_pred = 0 
    n = 0
    
    with torch.no_grad():
        model.eval()
        while len(dataset)>0:
            # get batch
            batch = get_batch(dataset,batch_size)
            normalize_batch(batch)
            X, y_true= convert_batch_to_tensors(batch)

            X = X.to(device)
            y_true = y_true.to(device)
            
            # Forward pass
            y_hat = model(X)
            _, predicted_labels1 = torch.max(y_prob1, 1)
            _, predicted_labels2 = torch.max(y_prob2, 1)


            n += y_true1.size(0) + y_true2.size(0)
            correct_pred += (predicted_labels1 == y_true1).sum() + (predicted_labels2 == y_true2).sum()

    return correct_pred.float() / n

def plot_losses(train_losses, valid_losses):
    '''
    Function for plotting training and validation losses
    '''
    
    # temporarily change the style of the plots to seaborn 
    plt.style.use('seaborn')

    train_losses = np.array(train_losses) 
    valid_losses = np.array(valid_losses)

    fig, ax = plt.subplots(figsize = (8, 4.5))

    ax.plot(train_losses, color='blue', label='Training loss') 
    ax.plot(valid_losses, color='red', label='Validation loss')
    ax.set(title="Loss over epochs", 
            xlabel='Epoch',
            ylabel='Loss') 
    ax.legend()
    fig.show()
    
    # change the plot style to default
    plt.style.use('default')
    