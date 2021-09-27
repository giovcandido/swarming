# This example was inspired by the following script: https://bit.ly/39E7gvK

import torch
import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split as split
from torch import optim
from torch.autograd import Variable

from swarming.utils.argument_parser import parse_arguments

from swarming.core.pso import PSO, ParallelPSO

# Load digit images from the dataset
digits = load_digits()

# X receives the data, Y the labels
X = digits.data
Y = digits.target

# Make train and validation sets
X_train, X_val, Y_train, Y_val = split(X, Y, test_size=0.5, random_state=42)

# Convert from numpy array to torch tensors
X_train = torch.from_numpy(X_train).float()
Y_train = torch.from_numpy(Y_train).long()
X_val = torch.from_numpy(X_val).float()

# Function to be optimized
def neural_network(arr):
    # Create an instance of the model
    model = torch.nn.Sequential()

    # Add first linear layer
    model.add_module('linear_1', torch.nn.Linear(64, 128, bias=False))

    # Add a sigmoid activation
    model.add_module('sigmoid_1', torch.nn.Sigmoid())

    # And a second linear layer
    model.add_module('linear_2', torch.nn.Linear(128, 10, bias=False))

    # Set batch size and epoch
    batch_size = 100
    epochs = 100

    # Gather parameters from argument array
    learning_rate = arr[0]
    momentum = arr[1]

    # Declare the loss function
    loss = torch.nn.CrossEntropyLoss(reduction='mean')

    # Declare the optimization algorithm
    opt = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # Perform training loop
    for _ in range(epochs):
        # Initial cost as 0.0
        cost = 0.0

        # Calculate the number of batches
        num_batches = len(X_train) // batch_size

        # For every batch
        for k in range(num_batches):
            # Declare start and end indexes for the current batch
            start, end = k * batch_size, (k + 1) * batch_size

            # Declare initial variables
            x = Variable(X_train[start:end], requires_grad=False)
            y = Variable(Y_train[start:end], requires_grad=False)

            # Reset the gradient
            opt.zero_grad()

            # Perform the foward pass
            fw_x = model.forward(x)
            output = loss.forward(fw_x, y)

            # Perform backward pass
            output.backward()

            # Update parameters
            opt.step()

            # Cost is the accumulated loss
            cost += output.item()

    # Declare validation variable
    x = Variable(X_val, requires_grad=False)

    # Perform backward pass with this variable
    output = model.forward(x)

    # Predict samples from evaluating set
    preds = output.data.numpy().argmax(axis=1)

    # Calculate accuracy
    acc = np.mean(preds == Y_val)

    # Return error rate to be minimized
    return 1 - acc

def main():
    # Parse arguments, such as parallel, swarm size etc.
    args = parse_arguments()

    parallel = args.parallel
    swarm_size = args.swarm_size
    iterations = args.iterations
    executions = args.executions

    # Function dimension
    dimension = 2

    # Now, let's set the bounds in order to restric the search space
    l_bounds = [0, 0]
    u_bounds = [1, 1]

    # Select the desired PSO class
    if not parallel:
        PSOClass = PSO
    else:
        PSOClass = ParallelPSO

    # Create a PSO instance
    pso = PSOClass(swarm_size, dimension, neural_network, l_bounds, u_bounds)

    # Run optimization task multiple times
    pso.optimize(iterations, executions)

if __name__ == '__main__':
    main()
