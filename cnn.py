#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 10:18:56 2020

@author: Lionel

Here we are interested in implementing a simple convolutional neural network using
Pytorch to perform a classification task on the standard MNIST data-set.
"""
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# import pickle

# file = open('mnist.pkl','rb')
# dat = pickle._Unpickler( file )
# dat.encoding = 'latin1'
# mnist = dat.load()


''' use torchvision to download MNIST,
 pre-process (convert to tensor and normalize) using transforms
 use the dataloader and split into train and test datasets'''
 
transform = transforms.Compose(
    [transforms.ToTensor(),
       transforms.Normalize((0.5,), (0.5,))])
mnist_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(mnist_trainset,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=2)

mnist_testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(mnist_testset,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=2)

''' MNIST has 10 classes '''
classes = (str(i) for i in range(10))

'''  display random images '''

import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(train_loader)
images, labels = dataiter.next()
# show images
imshow(torchvision.utils.make_grid(images))
print(labels)
# eee


''' create a convolutional neural network '''
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        ''' initialize and define the network layers, channels, and kernel sizes'''
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(3*3*64, 256)
        self.fc2 = nn.Linear(256, 10)


    def forward(self, x):
        ''' the forward pass does the computation by feeding each layer into the
        next while adding non-linear activation functions '''
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        x = x.view(x.shape[0],np.prod(x.shape[1:]))
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


net = Net()

import torch.optim as optim

''' Here we want to define our loss function, we could use mean squared error
but the cross entropy loss is more appropriate for classification tasks. Something
about the distribution of the error (ie. normal assumption for regression)'''
criterion = nn.CrossEntropyLoss()

''' Pick a standard optimizer, here stochastic gradient descent '''
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


''' collect loss data for visualization of the training'''
train_epoch_loss=[]
test_epoch_loss=[]
print_every = 500 # batches
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    test_loss = 0.0
    for i, data in enumerate(zip(train_loader,test_loader), 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0]

        ''' Initialize the gradient'''
        optimizer.zero_grad()

        # forward + backward + optimize
        ''' predict then compute the loss'''
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        ''' backpropagate the loss'''
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        
        
        ''' predict on the test set'''
        tinputs, tlabels = data[1]
        toutputs = net(tinputs)
        tloss = criterion(toutputs, tlabels)
        test_loss += tloss.item()


        if i % print_every == print_every - 1:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / print_every))
            train_epoch_loss.append(running_loss / print_every)
            test_epoch_loss.append(test_loss / print_every)
            running_loss = 0.0
            test_loss = 0.0

print('Finished Training')
xaxis = range(len(train_epoch_loss))
plt.figure()
plt.plot(xaxis,train_epoch_loss,c='red', label='train loss')
plt.plot(xaxis,test_epoch_loss,c='blue', label='test loss')
plt.scatter(xaxis,train_epoch_loss,c='red', s=5)
plt.scatter(xaxis,test_epoch_loss,c='blue', s=5)
plt.xlabel('batches')
plt.ylabel('avg loss per %s mini-batches' % print_every)
plt.legend()
plt.tight_layout()
