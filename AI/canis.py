import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
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

#Conviluted layers extrapolate details from the input and passes them to the fully connected layer
#Fully Connected Layer (Dense Layer) Takes the input and runs computations on them to generate output

Forward Propagation: Receive input data, process the information, and generate output
Backward Propagation: Calculate error and update the parameters of the network

Z = X * f
My Case:
image 128 x 128
filter 8 x 8

Example
#Convoluted layer
size 3 x 3 and a filter of size 2 x 2:
notice each equation has 4 problems. One for each cell in the 2x2 filter
(1x1 + 7x1 + 11x0 + 1x1) = 9
(7x1 + 2x1 + 1x0 + 23x1) = 32
(11x1 + 1x1 + 2x0 + 2x1) = 14
(1x1 + 23x1 + 2x0 + 2x1) = 26

Shaping the output 

Dimension of image = (n, n)
Dimension of filter = (f,f) 

Dimension of output will be ((n-f+1) , (n-f+1))

#Now the fully connected layer
The Linear transform equation. 
Z = WT.X + b
Where:
X = the input
W = is weight
b = is a constant

Lets look at the size of this convoluted network
(m, n) 
m is equal to the number of features or inputs of this layer
n will depend on the number of neurons in the layer. For instance
if we have 2 neurons then the shape of the weight matrix will be (4, 2)

Okay that was a lot now for backwards propigation

Lets give this a go with our params
input = 128x128
filter = 3x3

 
'''
rfh = logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%d-%m-%Y:%H:%M:%S',
    level=logging.DEBUG,
    filename='logs/logs.canisAI.log')

logger = logging.getLogger('my_app')

class Canis(nn.Module):
    def __init__(self):
        super(Canis, self).__init__()
        #self.convNN1
        '''
            For the vonv layers we need to specify three params
            in_channels, out_channels and kernel size(mxn)
            
            the linear layers require two params
            in_features and out_features
            
            In order to figure out the flattening multiplication we need this!
            outputSizeOfCov = [(inputSize + 2*pad - filterSize)/stride] + 1 
            (32 + 2 * 0 - 120)/1 + 1
            8 + 1
            9x9
        '''
        #Conv layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.pool = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        #Now the linear part
        '''
        in_features size of each input sample
        out_features size of each output sample
        Shaped
        in_features= (N, *, H_in)
        out_features= (N, *, H_out)
        '''
        self.fc1 = nn.Linear(54080, 3380)
        self.fc2 = nn.Linear(3380, 1690)
        self.fc3 = nn.Linear(1690, 845)
        self.fc4 = nn.Linear(845, 120)

    def forward(self, x):
        logger.debug("Information entering CNN -->")
        logger.debug(x)
        x = self.pool(F.relu(self.conv1(x)))
        logger.debug(x)
        x = self.pool(F.relu(self.conv2(x)))
        logger.debug(x)
        #Something is wrong with the pass from CNN to Dense Layers
        #Shape [1, 54080] Invalid for input of size 48672
        print("Information Exiting CNN -->")
        x = x.view(1, 54080)
        logger.debug(x)
        print("Information Entering Dense Layers")
        x = F.relu(self.fc1(x))
        logger.debug(x)
        x = F.relu(self.fc2(x))
        logger.debug(x)
        x = self.fc3(x)
        logger.debug(x)
        logger.debug("<--- Output")
        return x

canisAI = Canis()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(canisAI.parameters(), lr=0.001, momentum=0.9)