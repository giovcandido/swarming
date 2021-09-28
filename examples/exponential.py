from math import exp

from swarming.core.pso import PSO

# Function to be optimized
def exponential(x):
    # e^x - x
    return exp(x[0]) - x

def main():
    # Set the number of particles in the swarm
    swarm_size = 20

    # Set the number of function variables
    dimension = 1

    # Now, let's set the bounds in order to restric the search space
    lower_bound = [-10]
    upper_bound = [10]

    # Create a PSO instance
    pso = PSO(swarm_size, dimension, exponential, lower_bound, upper_bound)

    # Run optimization task multiple times
    pso.optimize(iterations=1000)

if __name__ == '__main__':
    main()
