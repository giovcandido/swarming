# ---------------------------------------------------------------------------- #
#                          Import the required modules                         #
# ---------------------------------------------------------------------------- #

import numpy as np

from swarming.utils.argument_parser import parse_arguments

from swarming.core.pso import PSO, ParallelPSO

# ---------------------------------------------------------------------------- #
#                            Problem definition part                           #
# ---------------------------------------------------------------------------- #

# Function dimension
dimension = 2

# Now, let's set the bounds in order to restric the search space
# Lower bounds
lower_bounds = np.full(dimension, -100)

# Upper bounds
upper_bounds = np.full(dimension, 100)

# Function to be optimized
def polynomial(x):
    return x[0] ** 2 + ((x[1] ** 2) / 16 - 5) ** 2 + 2 * x[0] + 6

# ---------------------------------------------------------------------------- #
#                 Main function, where the optimization occurs                 #
# ---------------------------------------------------------------------------- #

def main():
    # Parse arguments, such as parallel, swarm size etc.
    args = parse_arguments()

    # Select the desired PSO class
    if not args.parallel:
        PSOClass = PSO
    else:
        PSOClass = ParallelPSO

    # Create a PSO instance
    pso = PSOClass(args.swarm_size, dimension, polynomial, lower_bounds, upper_bounds)

    # Run optimization task multiple times
    pso.optimize(args.iterations, args.executions)

# ---------------------------------------------------------------------------- #
#                           Entry point of the script                          #
# ---------------------------------------------------------------------------- #

if __name__ == '__main__':
    main()