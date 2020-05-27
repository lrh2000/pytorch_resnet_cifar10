# This file is modified from https://pytorch.org/tutorials/beginner/fgsm_tutorial.html .
# Thanks for their work!

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

epsilons = [0, .05, .1, .15, .2, .25, .3]
pretrained_model2 = "save_resnet20/model.th"
pretrained_model = "pretrained_models/resnet20-12fca82f.th"
use_cuda = True
use_fake_grad = False
set_zero = False

# LeNet Model definition
import resnet
Net = resnet.resnet20

# MNIST Test dataset and dataloader declaration
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])),
        batch_size=1, shuffle=True)

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Initialize the network
model = torch.nn.DataParallel(Net()).to(device)
model2 = torch.nn.DataParallel(Net()).to(device)

# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location='cpu')['state_dict'])
model2.load_state_dict(torch.load(pretrained_model2, map_location='cpu')['state_dict'])

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()
model2.eval()

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad,data_grad2,set_zero = False):
    data_grad=data_grad.numpy()
    data_grad2=data_grad2.numpy()
    if set_zero:
        [a, b, c, d] = data_grad.shape
        for i in range(a):
            for j in range(b):
                for k in range(c):
                    for t in range(d):
                        if data_grad[i][j][k][t] < 0.001 or data_grad2[i][j][k][t]<0.001:
                            data_grad[i][j][k][t] = 0
    
    # Collect the element-wise sign of the data gradient
    sign_data_grad = torch.from_numpy(data_grad).cuda().sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    # perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def test(model, model2, device, test_loader, epsilon, set_zero=False):
    # Accuracy counter
    correct = correct2 = 0
    #adv_examples = []
    
    # Loop over all examples in test set
    for data, target in test_loader:
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)
        data2 = data.clone().detach().requires_grad_(True)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = not use_fake_grad

        # Forward pass the data through the model
        _, output = model(data)
        _, output2 = model2(data2)

        # init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # If the initial prediction is wrong, dont bother attacking, just move on
        #if init_pred.item() != target.item():
        #    continue

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
            data_grad = data.grad.data.cpu()
            data_grad2 = data2.grad.data.cpu()

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad,data_grad2,set_zero)

        # Re-classify the perturbed image
        _, output = model(perturbed_data)
        _, output2 = model2(perturbed_data)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        final_pred2 = output2.max(1, keepdim=True)[1] # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
        if final_pred2.item() == target.item():
            correct2 += 1
            
    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    final_acc2 = correct2/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {} v.s. {} / {} = {}".format(epsilon,
        correct, len(test_loader), final_acc, correct2, len(test_loader), final_acc2))
    
    # Return the accuracy and an adversarial example
    #return final_acc, adv_examples

#accuracies.append(acc)
#examples.append(ex)


# Run test for each epsilon
for eps in epsilons:
    test(model, model2, device, test_loader, eps, set_zero)
    #accuracies.append(acc)
    #examples.append(ex)

'''
plt.figure(figsize=(5,5))
plt.plot(epsilons, accuracies, "*-")
plt.yticks(np.arange(0, 1.1, step=0.1))
plt.xticks(np.arange(0, .35, step=0.05))
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.show()
# Plot several examples of adversarial samples at each epsilon
cnt = 0
plt.figure(figsize=(8,10))
for i in range(len(epsilons)):
    for j in range(len(examples[i])):
        cnt += 1
        plt.subplot(len(epsilons),len(examples[0]),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if j == 0:
            plt.ylabel("Eps: {}".format(epsilons[i]), fontsize=14)
        orig,adv,ex = examples[i][j]
        plt.title("{} -> {}".format(orig, adv))
        plt.imshow(ex, cmap="gray")
plt.tight_layout()
plt.show()
'''
