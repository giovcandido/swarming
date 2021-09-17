import numpy as np

from time import time

from modules.cli import parse_arguments

from modules.PSO import PSO
from modules.ParallelPSO import ParallelPSO

# ---------------------------------------------------------------------------- #
#                            Problem definition part                           #
# ---------------------------------------------------------------------------- #

# Function to be optimized
def function(x):    
    return x[0] ** 2 + ((x[1] ** 2) / 16 - 5) ** 2 + 2 * x[0] + 6

# Function dimension
dimension = 2

# Now, let's set the bounds in order to restric the search space
# Lower bounds
lower_bounds = np.full(dimension, -100)

# Upper bounds
upper_bounds = np.full(dimension, 100)

# ---------------------------------------------------------------------------- #
#                 Main function, where the optimization occurs                 #
# ---------------------------------------------------------------------------- #

def main():
    # Parse some arguments, such as if the execution should be in parallel, swarm size etc.
    args = parse_arguments()
    
    print('Running algorithm %i time(s)...\n' % args.times)

    # Run the PSO algorithm many times
    # It helps to check if the restricted search space is appropriate
    for i in range(1, args.times + 1):
        print('==> Run number %i' % i)

        # Defines a random seed to achieve constant results
        np.random.seed(i)

        # Create a PSO instance
        if args.parallel == 'n': 
            pso = PSO(args.swarm_size, dimension, function, lower_bounds, upper_bounds)
        else:
            pso = ParallelPSO(args.swarm_size, dimension, function, lower_bounds, upper_bounds)
        
        start_time = time()

        # Find an approximate solution with PSO
        approx_sol, fit = pso.optimize(args.iterations)

        end_time = time()

        print('Approximate solution:')

        for i, x in enumerate(approx_sol):
            x = round(x, 4) + 0
            print('x[%i] = %.4f' % (i, x))

        print('Fit = %.4f' % fit)
        print('Time spent: %f s\n' % (end_time - start_time))

# ---------------------------------------------------------------------------- #
#                           Entry point of the script                          #
# ---------------------------------------------------------------------------- #

if __name__ == '__main__':
    main()
