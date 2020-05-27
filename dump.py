# This file is modified from https://pytorch.org/tutorials/beginner/fgsm_tutorial.html .
# Thanks for their work!

from __future__ import print_function
import torch
from torchvision import datasets, transforms

pretrained_model = "save_cifar5-1_resnet20/model.th"
pretrained_model2 = "save_cifar5-2_resnet20/model.th"
#pretrained_model = "save_resnet20/model.th"
#pretrained_model2 = "pretrained_models/resnet20-12fca82f.th"
use_cuda = True
examples_limit = 10000

# LeNet Model definition
import resnet as resnet
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

model_data = torch.load(pretrained_model, map_location='cpu')
model2_data = torch.load(pretrained_model2, map_location='cpu')

# Initialize the network
model = torch.nn.DataParallel(Net(num_classes=model_data.get('num_classes', 10))).to(device)
model2 = torch.nn.DataParallel(Net(num_classes=model2_data.get('num_classes', 10))).to(device)

# Load the pretrained model
model.load_state_dict(model_data['state_dict'])
model2.load_state_dict(model2_data['state_dict'])

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

        i += 1
        if i >= examples_limit:
            break

    X = [torch.stack([X[j][i] for j in range(len(X))]) for i in range(len(X[0]))]
    Y = [torch.stack([Y[j][i] for j in range(len(Y))]) for i in range(len(Y[0]))]
    torch.save((X, Y), f'/tmp/model_tests.{examples_limit}.pt')

if __name__ == "__main__":
    test(model, model2, device, test_loader)
