import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

'''
Figuring out this convoluted layers and what not
So here is the equation

Nout = [Nin + 2p - k / s] + 1
Nin = The number of input features
Nout - Number of output features
k = convolution kernel size
p = convolution padding size
s = convolution stride size
'''

'''
All the images have been cleaned to 128x128
so I think we need 3, 4x4 (16 weight kernel)
'''

'''
Shapping the network :D
Suppose an image has size W x W, the filter has size F x F, the padding is P, and the stride is S. Then:
size = W x W
filter = F x F
padding = P
stride = S
the resulting equation
(W - F + 2P) / S + 1

Heres an example!
Images with a size of 100 x 100
A filter is 6 x 6
The padding is 7
and a stride of 4
the result of convolution
(100 - 6 + (2)(7)) / 4 + 1 = 28x28
(126 + 4) / 4 + 1 = 65x65

'''


'''
Okay we have made some progress on shaping the network but its
time for me to just understand wtf is going on here
Lets read this link and understand the math behind this!
https://www.analyticsvidhya.com/blog/2020/02/mathematics-behind-convolutional-neural-network/
'''
class Canis(nn.Module):
    def __init__(self):
        super(Canis, self).__init__()
        #self.convNN1
        self.conv1 = nn.Conv2d(3, 10, 64)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 16, 16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x, labels):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 10240)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

canisAI = Canis()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(canisAI.parameters(), lr=0.001, momentum=0.9)