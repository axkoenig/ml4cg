from __future__ import print_function
import torch
import sys

print(sys.version) # version of python i am using
print(sys.executable) # where is this version on my computer
print(torch.version)

# Tensors are similar to NumPy’s ndarrays, with the addition being that Tensors can also be used on a GPU to accelerate computing.
# Construct a 5x3 matrix, uninitialized: An uninitialized matrix is declared, 
# but does not contain definite known values before it is used. 
# When an uninitialized matrix is created, whatever values were in the allocated 
# memory at the time will appear as the initial values.
x = torch.empty(5, 3)
# print(x)

# Construct a 5x3 initialized matrix
x = torch.rand(5, 3)
# print(x)

# Construct a 5x3 matrix filled zeros and of dtype long
x = torch.zeros(5, 3, dtype=torch.long)
# print(x)

# Construct a tensor directly from data
x = torch.tensor([5.5, 3])
# print(x)

# create a tensor based on an existing tensor. These methods will reuse properties of the input tensor, e.g. dtype, unless new values are provided by user
x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes, tensors of ones
# print(x)

x = torch.randn_like(x, dtype=torch.float)    # override dtype!
# print(x)                                      # result has the same size

# get size of tensor
# print(x.size()) # torch.Size is in fact a tuple, so it supports all tuple operations.

# Resizing: If you want to resize/reshape tensor, you can use torch.view. The dimensions must be compatible, e.g. 2*8 can be reshaped to 4*4, but not to 3*5 = 15 elements
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
# print(x.size(), y.size(), z.size())

# Converting a Torch Tensor to a NumPy Array
a = torch.ones(5,3)
# print(a)

b = a.numpy()
# print(b)

a.add_(1) # The Torch Tensor and NumPy array will share their underlying memory locations (if the Torch Tensor is on CPU), and changing one will change the other.
# print(a)
# print(b) 

#  example of vector-Jacobian product
x = torch.randn(3, requires_grad=True)
print(x)

y = x * 2
while y.data.norm() < 1000: # norm: Returns the matrix norm or vector norm of a given tensor.
    y = y * 2

print(y)

# If you set its attribute .requires_grad as True, it starts to track all operations on it. 
# When you finish your computation you can call .backward() and have all the gradients computed automatically. 
# The gradient for this tensor will be accumulated into .grad attribute.
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)

print("Gradient of x:")
print(x.grad)

# You can also stop autograd from tracking history on Tensors with .requires_grad=True either 
# by wrapping the code block in with torch.no_grad()
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)

# Or by using .detach() to get a new Tensor with the same content but that does not require gradients
print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())

# usage of view function
# The view function is meant to reshape the tensor
a = torch.range(1, 16)
print(a)
a = a.view(4, 4)
print(a)

# What about -1 as a parameter to the function view?
# If there is any situation that you don't know how many rows you want but are sure of the number of columns, 
# then you can specify this with a -1. (Note that you can extend this to tensors with more dimensions. 
# Only one of the axis value can be -1). This is a way of telling the library: 
# "give me a tensor that has these many columns and you compute the appropriate number of rows 
# that is necessary to make this happen."