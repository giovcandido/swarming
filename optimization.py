import argparse
import time

import numpy as np

from modules.PSO import PSO


def function(x):    
    return x[0] ** 2 + ((x[1] ** 2) / 16 - 5) ** 2 + 2 * x[0] + 6


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--swarm-size', help="number of particles in the swarm", required=True)
parser.add_argument('-m', '--max-iterations', help="maximum number of iterations", required=True)
parser.add_argument('-t', '--times', help="number of times to run", required=True)


if __name__ == '__main__':
    # Parse swarm size and maximum number of iterations
    args = parser.parse_args()

    # Set the number of particles in the swarm
    swarm_size = int(args.swarm_size)

    # Set the dimension of the problem
    dimension = 2

    # Now, let's set the bounds in order to restric the search space
    # Lower bounds
    lower_bounds = np.full(dimension, -100)

    # Upper bounds
    upper_bounds = np.full(dimension, 100)

    # Number of times to run the algorithm
    times_to_run = int(args.times)

    print('Running algorithm %i time(s)...\n' % times_to_run)

    # Run the PSO algorithm many times
    # It helps to check if the restricted search space is appropriate
    for i in range(1, times_to_run + 1):
        print('==> Run number %i' % i)

        # Defines a random seed to achieve constant results
        np.random.seed(i)

        # Create a PSO instance
        pso = PSO(swarm_size, dimension, function, lower_bounds, upper_bounds)

        # Set maximum number of iterations
        max_iterations = int(args.max_iterations)
        
        start_time = time.time()

        # Find an approximate solution with PSO
        approx_sol, fit = pso.optimize(max_iterations)

        end_time = time.time()

        print('Approximate solution:')

        for i, x in enumerate(approx_sol):
            x = round(x, 4) + 0
            print('x[%i] = %.4f' % (i, x))

        print('Fit = %.4f' % fit)
        print('Time spent: %f s\n' % (end_time - start_time))
