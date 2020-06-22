import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.nn.Interpolate as Interpolate
import numpy as np


class UpDownMaxPooling(nn.Module):    
    def __init__(self):
        super(UpDownMaxPooling,self).__init__()
    
    def forward(self,x):
        for b in range(x.size()[0]):
            for i in range(x.size()[3]):
                # first elemnt in the column
                m = x[b][0][0][i]
                for j in range(x.size()[2]):
                    e = x[b][0][j][i]
                    if e > m:
                        m = e
                    else:
                        x[b][0][j][i] = m
        return x  

class DownUpMaxPooling(nn.Module):    
    def __init__(self):
        super(DownUpMaxPooling,self).__init__()
    
    def forward(self,x):
        for b in range(x.size()[0]):
            for i in range(x.size()[3]):
                # last element in the column
                m = x[b][0][x.size()[2]-1][i]
                for j in range(x.size()[2]-1,-1,-1):
                    e = x[b][0][j][i]
                    if e > m:
                        m = e
                    else:
                        x[b][0][j][i] = m
        return x  

class LeftRightMaxPooling(nn.Module):    
    def __init__(self):
        super(LeftRightMaxPooling,self).__init__()
    
    def forward(self,x):
        for b in range(x.size()[0]):
            for i in range(x.size()[2]):
                # first elemnt in the row
                m = x[b][0][i][0]
                for j in range(x.size()[3]):
                    e = x[b][0][i][j]
                    if e > m:
                        m = e
                    else:
                        x[b][0][i][j] = m
        return x 


class RightLeftMaxPooling(nn.Module):    
    def __init__(self):
        super(RightLeftMaxPooling,self).__init__()
    
    def forward(self,x):
        for b in range(x.size()[0]):
            for i in range(x.size()[2]):
                # last elemnt in the row
                m = x[b][0][i][x.size()[2]-1]
                for j in range(x.size()[3]-1,-1,-1):
                    e = x[b][0][i][j]
                    if e > m:
                        m = e
                    else:
                        x[b][0][i][j] = m
        return x 

class CornerPooling(nn.Module):    
    def __init__(self,t):
        super(CornerPooling,self).__init__()
        '''
        t = 0 for top left corner
        t = 1 for bottom right corner
        '''    
        self.t = t
        self.rl = RightLeftMaxPooling()
        self.lr = LeftRightMaxPooling()
        self.td = UpDownMaxPooling()
        self.bu = DownUpMaxPooling()

        self.conv3x3_bn_r = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1,padding = 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )

        self.conv3x3_bn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1,padding = 1),
            nn.BatchNorm2d(1),
        )

        self.conv3x3_r = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1,padding = 1),
            nn.ReLU(),
        )
        self.conv1x1_bn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1),
        )

        self.relu = nn.ReLU()

    def forward(self,x):
        x1 = self.conv3x3_bn_r(x)
        x2 = self.conv3x3_bn_r(x)
        if self.t == 0:
            i1 = self.rl(x1)
            i2 = self.bu(x2)
        else:
            i1 = self.lr(x1)
            i2 = self.td(x2)
        b1 = i1 + i2
        b1 = self.conv3x3_bn(b1)
        b2 = self.conv1x1_bn(x)
        Sum = b1 + b2
        Sum_r = self.relu(Sum)
        out = self.conv3x3_bn_r(Sum_r)
        out = self.conv3x3_r(out)
        out = self.conv1x1(out)

        return out

# i = np.zeros((2,1,256,256))
# i = np.random.rand(2,1,3,3)
# j = np.random.rand(2,1,3,3)
# t = torch.from_numpy(i).float()
# g = torch.from_numpy(j).float()
# print(t)
# m = CornerPooling(0).float()
# print(m(t))