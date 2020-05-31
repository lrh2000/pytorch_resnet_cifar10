# This file is modified from https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents .
# Thanks for their work!

# library
# standard library
import os

# third-party library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 50               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 512
LR = 0.001              # learning rate
DOWNLOAD_MNIST = False


# Mnist digits dataset
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,
)

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# pick 2000 samples to speed up testing
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_x = test_x.view(test_x.size(0), -1)
test_y = test_data.targets
best_acc = 0


def InitParams(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_output): # n_feature=28*28  n_output=10
        super(Net, self).__init__()
        self.hidden0 = torch.nn.Linear(n_feature, 8*n_feature)   # hidden layer
        self.hidden1 = torch.nn.Linear(8*n_feature, 4096)  # hidden layer
        self.hidden2 = torch.nn.Linear(4096, 1024)  # hidden layer
        self.hidden3 = torch.nn.Linear(1024, 256)  # hidden layer
        self.predict = torch.nn.Linear(256, n_output)   # output layer

    def forward(self, x):
        middle = [x]
        x = F.relu(self.hidden0(x))      # activation function for hidden layer
        middle.append(x)
        x = F.relu(self.hidden1(x))      # activation function for hidden layer
        middle.append(x)
        x = F.relu(self.hidden2(x))  # activation function for hidden layer
        middle.append(x)
        x = F.relu(self.hidden3(x))  # activation function for hidden layer
        middle.append(x)
        output = self.predict(x)             # linear output
        middle.append(output)
        return middle, output


net = Net(28*28, 10)
# net2 = Net(28*28, 10)
# InitParams(net2)
# print(net)  # net architecture

optimizer = torch.optim.Adam(net.parameters(), lr=LR)   # optimize all cnn parameters
# optimizer = torch.optim.Adam(net2.parameters(), lr=LR)   # optimize all cnn parameters
# optimizer = torch.optim.Adam(cnn2.parameters(), lr=LR)   # optimize all cnn2 parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
        b_x = b_x.view(b_x.size(0), -1)
        output = net(b_x)[1]               # cnn output
        # output = net2(b_x)[1]               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if step % 50 == 0:
            pred_y = torch.max(output, 1)[1].data.numpy()
            accuracy = float((pred_y == b_y.data.numpy()).astype(int).sum()) / float(b_y.size(0))
            print('[Train] Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(),
                  '| train accuracy: %.4f' % accuracy)

    test_output = net(test_x)[1]
    # test_output = net2(test_x)[1]
    pred_y = torch.max(test_output, 1)[1].data.numpy()
    loss = loss_func(test_output, test_y)  # cross entropy loss
    accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
    if accuracy > best_acc:
        best_acc = accuracy
    print('[Test] Epoch: ', epoch, '| test loss: %.4f' % loss.data.numpy(), '| test accuracy: %.4f' % accuracy)

torch.save(net.state_dict(), './save_linear/net.pkl')
# torch.save(net2.state_dict(), './save_linear/net2.pkl')
