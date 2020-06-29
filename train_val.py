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

import skimage

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
        print(X.size())
        print(y_true.size())

        optimizer.zero_grad()


        # Forward pass
        y_hat = model(X)
        
        # loss
        cross_entropy_loss = cross_entropy_loss_criterion(y_hat, y_true) 
        
        epoch_loss += cross_entropy_loss.item()
        
        i += 1        
        print("batch: ",i,"done")
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

    # plot_losses(train_losses, valid_losses)
    heat_map (model,device)
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

def heat_map (model,device):
    # # generate img
    # # randomize background color
    # img_size = 256
    # r = random.randint(0,255)
    # g = random.randint(0,255)
    # b = random.randint(0,255)
    # back_color = [r,g,b]

    # img = np.ones((img_size,img_size,3)).astype(int)*back_color

    # # forground
    # # randomize foreground color
    # r = random.randint(0,255)
    # g = random.randint(0,255)
    # b = random.randint(0,255)
    # fore_color = [r,g,b]
    # # randomize center
    # center_x = random.randint(img_size//8,(7 * img_size)//8)
    # center_y = random.randint(img_size//8,(7 * img_size)//8)
    # radius = random.randint(img_size//10,np.min([img_size - 1 - center_x, img_size - 1 - center_y, center_x, center_y]))

    # rr,cc = skimage.draw.circle(center_x, center_y, radius)
    # img[rr,cc] = fore_color

    img = np.array(plt.imread("images/0.jpg"))
    imgs = []
    imgs.append(img)
    imgs = np.moveaxis(imgs, -1, 1)
    # imgs = torch.Tensor(imgs)

    t = torch.from_numpy(imgs)
    t = t.to(device)
    # t = t.type(torch.DoubleTensor)
    # print(t)
    model = model.float()
    u = model(t.float())
    h1 = u[0][0].cpu().detach().numpy()
    h2 = u[0][1].cpu().detach().numpy()
    
    
    plt.imshow(img)
    plt.show()
    plt.imshow(h1)
    plt.show()
    plt.imshow(h2)
    plt.show()

    with open('bb.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            path = row[0]
            x1 = int(row[1])
            y1 = int(row[2])
            x2 = int(row[3])
            y2 = int(row[4])
            img1 = np.zeros((256,256))
            img1[x1][y1] = 1.0
            img2 = np.zeros((256,256))
            img2[x2][y2] = 1.0
            label = np.zeros((2,256,256))
            label[0] = img1
            label[1] = img2
            plt.imshow(img1)
            plt.show()
            plt.imshow(img2)
            plt.show()
            break
