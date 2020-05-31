# This file is modified from https://pytorch.org/tutorials/beginner/fgsm_tutorial.html .
# Thanks for their work!

from __future__ import print_function
import torch
from torchvision import datasets, transforms
import torch.nn as nn

pretrained_model = "./save_cnn/cnn.pkl"
pretrained_model2 = "./save_cnn/cnn2.pkl"
use_cuda = True
examples_limit = 10000

# LeNet Model definition
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
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        middle.append(x)
        return middle, output


# MNIST Test dataset and dataloader declaration
test_loader = torch.utils.data.DataLoader(datasets.MNIST(root='./mnist/', train=False, transform=transforms.ToTensor()), batch_size=1, shuffle=True)

# Define what device we are using
print("CUDA Available: ", torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Initialize the network
model = CNN().to(device)
model2 = CNN().to(device)

# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location='cuda')['state_dict'])
model2.load_state_dict(torch.load(pretrained_model2, map_location='cuda')['state_dict'])

# Set the model in evaluation mode. In this case this is for the Dropout layers
model.eval()
model2.eval()


def test(model, model2, device, test_loader):
    X, Y = [], []
    i = 0

    # Loop over all examples in test set
    for data, target in test_loader:
        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Forward pass the data through the model
        middle, output = model(data)
        middle = [x.detach().view(-1) for x in middle]
        X.append(middle)

        # Re-classify the perturbed image
        middle2, output2 = model2(data)
        middle2 = [y.detach().view(-1) for y in middle2]
        Y.append(middle2)

        print(i)
        i += 1
        if i >= examples_limit:
            break

    X = [torch.stack([X[j][i] for j in range(len(X))]) for i in range(len(X[0]))]
    Y = [torch.stack([Y[j][i] for j in range(len(Y))]) for i in range(len(Y[0]))]
    print('----')
    torch.save((X, Y), f'./tmp/cnn_model_tests.{examples_limit}.pt')

if __name__ == "__main__":
    test(model, model2, device, test_loader)
