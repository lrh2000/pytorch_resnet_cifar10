# This file is modified from https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents .
# Thanks for their work!

# library
# standard library
import os

# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 50               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 512
LR = 0.001              # learning rate
DOWNLOAD_MNIST = False
best_acc = 0


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
test_y = test_data.targets


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


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        middle = [x]
        x = self.conv1(x)
        middle.append(x)
        x = self.conv2(x)
        middle.append(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        middle.append(x)
        return middle, output    # return x for visualization


cnn = CNN()
# cnn2 = CNN()
# InitParams(cnn2)
# print(cnn)  # net architecture

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
# optimizer = torch.optim.Adam(cnn2.parameters(), lr=LR)   # optimize all cnn2 parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader

        output = cnn(b_x)[1]               # cnn output
        # output = cnn2(b_x)[1]               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if step % 50 == 0:
            pred_y = torch.max(output, 1)[1].data.numpy()
            accuracy = float((pred_y == b_y.data.numpy()).astype(int).sum()) / float(b_y.size(0))
            print('[Train] Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| train accuracy: %.4f' % accuracy)

    test_output = cnn(test_x)[1]
    # test_output = cnn2(test_x)[1]
    pred_y = torch.max(test_output, 1)[1].data.numpy()
    loss = loss_func(test_output, test_y)  # cross entropy loss
    accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
    if accuracy > best_acc:
        best_acc = accuracy
    print('[Test] Epoch: ', epoch, '| test loss: %.4f' % loss.data.numpy(), '| test accuracy: %.4f' % accuracy)

torch.save({
            'state_dict': cnn.state_dict(),
            'best_prec1': best_acc,
        }, './save_cnn/cnn.pkl')
# torch.save({
#             'state_dict': cnn2.state_dict(),
#             'best_prec1': best_acc,
#         }, './save_cnn/cnn2.pkl')

# torch.save(cnn.state_dict(), './save_cnn/cnn.pkl')
# torch.save(cnn2.state_dict(), './save_cnn/cnn2.pkl')
