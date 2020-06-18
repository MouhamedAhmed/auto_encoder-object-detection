import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.nn.Interpolate as Interpolate
import numpy as np

class Interpolate(nn.Module):
    def __init__(self):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        
    def forward(self, x):
        x = self.interp(x, mode='bilinear', scale_factor=2)
        return x



class Autoencoder(nn.Module):    
    def __init__(self):
        super(Autoencoder,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=6, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1),
            nn.ReLU(),
            )

        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=1),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5, stride=1),
            nn.ReLU(),

            Interpolate(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, stride=1),
            nn.ReLU(),

            Interpolate(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=6, stride=1),
            nn.ReLU(),

            Interpolate(),
            nn.ConvTranspose2d(in_channels=16, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(),

            Interpolate(),
            nn.ConvTranspose2d(in_channels=6, out_channels=2, kernel_size=5, stride=1),
            nn.ReLU(),
            )
        
        self.sig = nn.Sigmoid()

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.sig(x)
        return x

# m = Autoencoder()
# i = np.zeros((1,3,256,256))
# t = torch.from_numpy(i)
# # t = t.type(torch.DoubleTensor)
# # print(t)
# m = m.float()
# u = m(t.float())
# print(u.size())