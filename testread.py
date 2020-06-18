import csv
import numpy as np

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

    training = dataset[0:int(0.8*len(dataset))]
    validation = dataset[int(0.8*len(dataset)):int(0.9*len(dataset))]
    testing = dataset[int(0.9*len(dataset)):len(dataset)]
    return training,validation,testing



load_data()
