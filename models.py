## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # the output Tensor for one image, will have the dimensions: (224-5)/1 + 1 = (32, 220, 220)
        # after one pool layer, this will become (32, 110, 110)
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
                
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        
        # second conv layer: 32 inputs, 64 outputs, 5x5 conv
        ## output size = (W-F)/S +1 = (110-5)/1 +1 = 106
        # the output tensor will have dimensions: (16, 106, 106)
        # after another pool layer this will become (16, 53, 53)
        self.conv2 = nn.Conv2d(32, 64, 5)
        
        # third conv layer: 64 inputs, 128 outputs, 5x5 conv
        ## output size = (W-F)/S +1 = (53-5)/1 +1 = 49
        # the output tensor will have dimensions: (128, 49, 49)
        # after another pool layer this will become (128, 24, 24)
        self.conv3 = nn.Conv2d(64, 128, 5)
        
        # fourth conv layer: 128 inputs, 512 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (24-3)/1 +1 = 22
        # the output tensor will have dimensions: (512, 22, 22)
        # after another pool layer this will become (512, 11, 11)
        self.conv4 = nn.Conv2d(128, 512, 3)
        
        
        # 512 outputs * the 11,11 filtered/pooled map size
        self.fc1 = nn.Linear(512*11*11, 1024)
        
        # dropout with p=0.4
        self.fc1_drop = nn.Dropout(p=0.4)
        
        # finally, create 10 output channels (2 for each of the 68 keypoint (x, y) pairs)
        self.fc2 = nn.Linear(1024, 136)

        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        # two conv/relu + pool layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        
        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
