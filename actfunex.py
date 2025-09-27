import torch
import torchvision
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torch.optim as optim



if __name__ == "__main__":
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')

    else:
        DEVICE = torch.device('cpu')

    print("Using PyTorch version:", torch.__version__,' Device:', DEVICE)

    BATCH_SIZE = 32
    EPOCHS = 10
    dataset = MNIST(root='data/',download =True,transform = ToTensor())
    #Splitting dataset to train and validaion

    val_size = 10000

    train_size = len(dataset) - val_size

    train_set,val_set = random_split(dataset,[train_size,val_size])
    batch_size = 128

    train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(val_set,batch_size=batch_size,shuffle=True,num_workers=2,pin_memory=True)
    class Net(nn.Module):
    #This defines the structure of the NN.
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=7,  stride=2, padding=3)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=7,  stride=2, padding=3)
            self.conv3 = nn.Conv2d(20, 20, kernel_size=7,  stride=1, padding=3)
            self.conv4 = nn.Conv2d(20, 20, kernel_size=7,  stride=1, padding=3)
            self.conv5 = nn.Conv2d(20, 20, kernel_size=7,  stride=1, padding=3)
            self.conv6 = nn.Conv2d(20, 20, kernel_size=7,  stride=1, padding=3)
            self.conv7 = nn.Conv2d(20, 20, kernel_size=7,  stride=1, padding=3)
            self.conv8 = nn.Conv2d(20, 20, kernel_size=7,  stride=1, padding=3)
            self.fc1 = nn.Linear(980, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):

            x = act(self.conv1(x),epoch)
            x = act(self.conv2(x),epoch)

            h = act(self.conv3(x),epoch)
            x = act(self.conv4(h)+x,epoch)

            h = act(self.conv5(x),epoch)
            x = act(self.conv6(h)+x,epoch)

            h = act(self.conv7(x),epoch)
            x = act(self.conv8(h)+x,epoch)

            b,c,h,w = x.shape

            x = x.view(-1, c*h*w)

            #Fully Connected Layer/Activation

            x = act(self.fc1(x),epoch)
            #Fully Connected Layer/Activation
            x = self.fc2(x)
            #Softmax gets probabilities.
            return F.log_softmax(x, dim=1)


    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if args['cuda']:
                data, target = data.cuda(), target.cuda()
            #Variables in Pytorch are differenciable.
            #This will zero out the gradients for this batch.
            optimizer.zero_grad()
            output = model(data)
            # Calculate the loss The negative log likelihood loss. It is useful to train a classification problem with C classes.
            loss = F.nll_loss(output, target)
            #dloss/dx for every Variable
            loss.backward()
            #to do a one-step update on our parameter.
            optimizer.step()

            #Print out the loss periodically.
            if batch_idx % args['log_interval'] == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss))
    model = Net()

    def test():
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                if args['cuda']:
                    data, target = data.cuda(), target.cuda()
                data, target = data, target
                output = model(data)
                test_loss += F.nll_loss(output, target, size_average=False) # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

            test_loss /= len(test_loader.dataset)

            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
    def act(x, epoch):
        return F.sigmoid(x)+x*0.9120**(epoch)    
    
    args={}
    kwargs={}
    args['batch_size']=1000
    args['test_batch_size']=1000
    args['epochs']=100  #The number of Epochs is the number of times you go through the full dataset.
    args['lr']=0.01 #Learning rate is how fast it will decend.
    args['momentum']=0.5 #SGD momentum (default: 0.5) Momentum is a moving average of our gradients (helps to keep direction).

    args['seed']=1 #random seed
    args['log_interval']=110
    args['cuda']=False
    if args['cuda']:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])
    for epoch in range(1, args['epochs'] + 1):
        train(epoch)
        test()

    
        