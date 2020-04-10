import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

'''
A typical training procedure for a neural network is as follows:

    Define the neural network that has some learnable parameters (or weights)
    Iterate over a dataset of inputs
    Process input through the network
    Compute the loss (how far is the output from being correct)
    Propagate gradients back into the network’s parameters
    Update the weights of the network, typically using a simple update rule: weight = weight - learning_rate * gradient
'''

# Define the neural network that has some learnable parameters (or weights)
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from convoluted image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # You just have to define the forward function, and the backward function (where gradients are computed) 
    # is automatically defined for you using autograd. 
    # You can use any of the Tensor operations in the forward function.
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # to be able to go from a convolutional layer to a fully connected layer features need to be flattened
        # so we want it to reshape this array to have the right amount of columns, and let it decide the number of rows itself
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x # output at the end of the network

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

# The learnable parameters of a model are returned by net.parameters()
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight
print(params[1].size())  # bias
print(params[2].size())  # conv2's .weight
print(params[3].size())  # bias
print(params[4].size())  # fc1's .weight
print(params[5].size())  # fc1's .weight (bias)
print(params[6].size())  # fc2's .weight
print(params[7].size())  # fc1's .weight (bias)
print(params[8].size())  # fc3's .weight
print(params[9].size())  # fc1's .weight (bias)

# we input an image. This LeNet expects input 32x32
input = torch.randn(1, 1, 32, 32)
out = net(input)

# Zero the gradient buffers of all parameters and backprops with random gradients 
# (because we don't know the actual desired output yet)
net.zero_grad()
out.backward(torch.randn(1, 10))

# loss function: A loss function takes the (output, target) pair of inputs, 
# and computes a value that estimates how far away the output is from the target.
# L1, MSE, cross-entropy, CTC, NLL, and the list goes on
output = net(input)
print(output)
target = torch.randn(10)  # a dummy target, for example for 10 classes of images
print("target")
print(target)
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

# Backprop
# To backpropagate the error all we have to do is to loss.backward(). 
# You need to clear the existing gradients though, else gradients will be accumulated to existing gradients.

net.zero_grad() # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward() # backpropagate the error

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# The only thing left to learn is: Updating the weights of the network with e.g. SGD (stoch. gradient descent)
# weight = weight - learning_rate * gradient
learning_rate = 0.01
for f in net.parameters(): # update all parameters of the net
    f.data.sub_(f.grad.data * learning_rate) # in place operation

# However, as you use neural networks, you want to use various different update rules such as SGD, 
# Nesterov-SGD, Adam, RMSProp, etc. To enable this, we built a small package: torch.optim that implements 
# all these methods. Using it is very simple: import torch.optim as optim

# create your optimizer, e.g. SGD
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop
optimizer.zero_grad() # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step() # optimizer does the update

# Specifically for vision, we have created a package called torchvision, that has data loaders for 
# common datasets such as Imagenet, CIFAR10, MNIST, etc. and data transformers for images, viz., 
# torchvision.datasets and torch.utils.data.DataLoader.