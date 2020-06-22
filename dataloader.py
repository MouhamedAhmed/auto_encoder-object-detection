import os
import numpy as np
import random
import copy
from PIL import Image

import torch
import matplotlib.pyplot as plt
import random
import csv

def load_data():
    dataset = []
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
            entry = {
                "path": path,
                "label": label
            }
            dataset.append(entry)

    # training = dataset[0:int(0.8*len(dataset))]
    # validation = dataset[int(0.8*len(dataset)):int(0.9*len(dataset))]
    # testing = dataset[int(0.9*len(dataset)):len(dataset)]
    training = [dataset[0]]
    validation = [dataset[0]]
    testing = [dataset[0]]
    return training,validation,testing


################

def get_batch (dataset,batch_size):
    '''
    dataset: list of lists each containig data of a label
    '''
    if len(dataset) == 0:
        return []
    batch_size = min(batch_size,len(dataset))
    
    # get random indices = batch_size at max
    indices = random.sample(range(0, len(dataset)), batch_size)

    # get paths
    batch = []
    indices.sort(reverse=True) 
    for i in range (len(indices)):
        c = copy.deepcopy(dataset[indices[i]])
        path = c["path"]
        # load the image
        image = Image.open(path)
        if len(list(np.array(image).shape)) != 3:
            image = image.convert('RGB')

        image = np.asarray(image)
        
        d = {
            "path": c["path"],
            "image": image,
            "label": c["label"]
        }
        batch.append(d)
        del dataset[indices[i]]

    random.shuffle(batch)
    return batch

##################
def normalize_batch(batch):
    for i in batch:
        i["image"] =  i["image"]/255

##################
def convert_batch_to_tensors(batch):
    batch_size = len(batch)
    images = []
    labels = []
    for i in batch:
        image = i["image"]
        label = i["label"]

        images.append(image)
        labels.append(label)


    images = np.asarray(images)
    labels = np.asarray(labels)
    images = np.moveaxis(images, -1, 1)
    # labels = np.reshape(labels,(labels.shape[0],1))
    images = torch.Tensor(images)
    labels = torch.LongTensor(labels).float()
    
    return images,labels

