# This script is based on one of example scripts in the Opytimizer project repository

# This example script can be found in the following link: 
# https://github.com/gugarosa/opytimizer/blob/master/examples/integrations/pytorch/neural_network.py

# ---------------------------------------------------------------------------- #
#                          Import the required modules                         #
# ---------------------------------------------------------------------------- #

import torch
import numpy as np

from time import time
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import optim
from torch.autograd import Variable

from modules.cli import parse_arguments

from utils.log_creator import create_logger

from modules.PSO import PSO
from modules.ParallelPSO import ParallelPSO

# ---------------------------------------------------------------------------- #
#                            Problem definition part                           #
# ---------------------------------------------------------------------------- #

# Function dimension
dimension = 2

# Now, let's set the bounds in order to restric the search space
lower_bounds = [0, 0]
upper_bounds = [1, 1]

# ----------------------------- Data preparation ----------------------------- #

# Load digit images from the dataset
digits = load_digits()

# X receives the data 
X = digits.data

# Y receives the labels
Y = digits.target

# Split data/label in train and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.5, random_state=42)

# Convert from numpy array to torch tensors
X_train = torch.from_numpy(X_train).float()
X_val = torch.from_numpy(X_val).float()
Y_train = torch.from_numpy(Y_train).long()

# --------------------------------- Function --------------------------------- #

# Function to be optimized
def neural_network(arr):
    # Create an instance of the model
    model = torch.nn.Sequential()

    # Set some parameters
    n_features = 64
    n_hidden = 128
    n_classes = 10

    # Add first linear layer
    model.add_module("linear_1", torch.nn.Linear(n_features, n_hidden, bias=False))

    # Add a sigmoid activation
    model.add_module("sigmoid_1", torch.nn.Sigmoid())

    # And a second linear layer
    model.add_module("linear_2", torch.nn.Linear(n_hidden, n_classes, bias=False))

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

# ---------------------------------------------------------------------------- #
#                 Main function, where the optimization occurs                 #
# ---------------------------------------------------------------------------- #

def main():
    # Parse some arguments, such as if the execution should be in parallel, swarm size etc.
    args = parse_arguments()

    logger = create_logger('logs', 'neural_net.log')

    logger.info('Running algorithm %i time(s)...\n' % args.times)

    # Run the PSO algorithm many times
    # It helps to check if the restricted search space is appropriate
    for i in range(1, int(args.times) + 1):
        logger.info('==> Run number %i' % i)

        # Defines a random seed to achieve constant results
        np.random.seed(i)

        # Creates the space, optimizer and function
        if not args.parallel: 
            pso = PSO(args.swarm_size, dimension, neural_network, lower_bounds, upper_bounds)
        else:
            pso = ParallelPSO(args.swarm_size, dimension, neural_network, lower_bounds, upper_bounds)

        start_time = time()

        # Find an approximate solution with PSO
        approx_sol, fit = pso.optimize(args.iterations)

        end_time = time()

        logger.info('Approximate solution:')

        for i, x in enumerate(approx_sol):
            x = round(x, 16) + 0
            logger.info('x[%i] = %.16f' % (i, x))

        logger.info('Fit = %.16f' % fit)
        logger.info('Acc = %.16f\n' % ((1 - fit) * 100))
        logger.info('Time spent: %f s\n' % (end_time - start_time))

if __name__ == '__main__':
    main()
