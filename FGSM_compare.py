# This file is modified from https://pytorch.org/tutorials/beginner/fgsm_tutorial.html .
# Thanks for their work!

from __future__ import print_function
import resnet
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt


pretrained_model = "save_resnet20/model.th"
pretrained_model2 = "pretrained_models/resnet20-12fca82f.th"
use_cuda = True
use_fake_grad = False
set_zero = False # set grad if grad<0.001 to 0

# LeNet Model definition
Net = resnet.resnet20

# MNIST Test dataset and dataloader declaration
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])),
    batch_size=1, shuffle=True)

# Define what device we are using
print("CUDA Available: ", torch.cuda.is_available())
device = torch.device("cuda" if (
    use_cuda and torch.cuda.is_available()) else "cpu")

# Initialize the network
model = torch.nn.DataParallel(Net()).to(device)
model2 = torch.nn.DataParallel(Net()).to(device)


# Load the pretrained model
model.load_state_dict(torch.load(
    pretrained_model, map_location='cpu')['state_dict'])
model2.load_state_dict(torch.load(
    pretrained_model2, map_location='cpu')['state_dict'])

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()
model2.eval()

# FGSM attack code


def cal(x, y):
    sum_x = sum_y = 0
    total_and = 0
    diff_and = 0
    total_or = 0
    diff_or = 0
    x_sign = np.sign(x)
    y_sign = np.sign(y)
    [a, b, c, d] = x.shape
    for i in range(a):
        for j in range(b):
            for k in range(c):
                for t in range(d):

                    if x[i][j][k][t] < 0.001 and y[i][j][k][t] < 0.001:
                        total_and += 1
                        if x_sign[i][j][k][t] != y_sign[i][j][k][t]:
                            diff_and += 1

                    if x[i][j][k][t] < 0.001 or y[i][j][k][t] < 0.001:
                        total_or += 1
                        if x_sign[i][j][k][t] != y_sign[i][j][k][t]:
                            diff_or += 1

                    if x[i][j][k][t] < 0.001:
                        sum_x += 1
                        x[i][j][k][t] = 0
                    if y[i][j][k][t] < 0.001:
                        sum_y += 1
                        y[i][j][k][t] = 0
    # print(sum_x, "  ", sum_y, "  ", total_and, "  ",
    #       diff_and, "  ", total_or, "  ", diff_or)
    return torch.from_numpy(x), torch.from_numpy(y), sum_x, sum_y, total_and, diff_and, total_or, diff_or


def sign_compare(data_grad, data_grad2):
    sign_data_grad = data_grad.sign()
    sign_data_grad2 = data_grad2.sign()

    difference = torch.abs(sign_data_grad-sign_data_grad2)

    return torch.sum(difference == 1), torch.sum(difference == 2)


def test(model, model2, device, test_loader, set_zero=False):
    # Accuracy counter
    correct = correct2 = 0
    #adv_examples = []
    sign_sum_1 = 0
    sign_sum_2 = 0
    cnt = 0

    with open("record.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Loop over all examples in test set
        for data, target in test_loader:
            cnt += 1
            print(cnt)
            # Send the data and label to the device
            data, target = data.to(device), target.to(device)

            data2 = data.clone().detach().requires_grad_(True)

            # Set requires_grad attribute of tensor. Important for Attack
            data.requires_grad = not use_fake_grad

            # Forward pass the data through the model
            _, output = model(data)
            _, output2 = model2(data2)

            if use_fake_grad:
                data_grad = torch.rand_like(data) - 0.5
            else:
                # Calculate the loss
                loss = F.nll_loss(output, target)
                loss2 = F.nll_loss(output2, target)

                # Zero all existing gradients
                model.zero_grad()
                model2.zero_grad()

                # Calculate gradients of model in backward pass
                loss.backward()
                loss2.backward()

                # Collect datagrad
                data_grad = data.grad.data
                data_grad2 = data2.grad.data

                if set_zero:
                    data_grad, data_grad2, sum_x, sum_y, total_and, diff_and, total_or, diff_or = cal(
                        data_grad.cpu().numpy(), data_grad2.cpu().numpy())
                    writer.writerow([sum_x, sum_y, total_and, diff_and, total_or, diff_or])

                a, b = sign_compare(data_grad, data_grad2)
                sign_sum_1 += a + b
                sign_sum_2 += b

        if set_zero:
            print("sign1_similar:", 1-sign_sum_1.item()/(3*32*32*10000))
            print("sign2_similar:", 1-sign_sum_2.item()/(3*32*32*10000))
        else:
            print("sign_similar:", 1-sign_sum_1.item()/(3*32*32*10000))


# Run test for each epsilon
test(model, model2, device, test_loader, set_zero)
