# This file is modified from https://pytorch.org/tutorials/beginner/fgsm_tutorial.html .
# Thanks for their work!

from __future__ import print_function
import torch
from torchvision import datasets, transforms
import torch.nn.functional as F

pretrained_model = "./save_linear/net.pkl"
pretrained_model2 = "./save_linear/net2.pkl"
use_cuda = True
examples_limit = 5000

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


# MNIST Test dataset and dataloader declaration
test_data = datasets.MNIST(root='./mnist/', train=False)
test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_x = test_x.view(test_x.size(0), -1)
test_y = test_data.targets
test_dataset = torch.utils.data.TensorDataset(test_x, test_y)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

# Define what device we are using
print("CUDA Available: ", torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# Initialize the network
model = Net(28*28, 10).to(device)
model2 = Net(28*28, 10).to(device)

# Load the pretrained model
model.load_state_dict(torch.load(pretrained_model, map_location='cuda'))
model2.load_state_dict(torch.load(pretrained_model2, map_location='cuda'))

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
    torch.save((X, Y), f'./tmp/linear_model_tests.{examples_limit}.pt')

if __name__ == "__main__":
    test(model, model2, device, test_loader)
