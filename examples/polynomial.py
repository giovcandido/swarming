from swarming.core.pso import PSO

# Function to be optimized
def polynomial(x):
    return x[0] ** 2 + ((x[1] ** 2) / 16 - 5) ** 2 + 2 * x[0] + 6

def main():
    # Set the number of particles in the swarm
    swarm_size = 20

    # Set the number of function variables
    dimension = 2

    # Now, let's set the bounds in order to restric the search space
    l_bounds = [-100, -100]
    u_bounds = [100, 100]

    # Create a PSO instance
    pso = PSO(swarm_size, dimension, polynomial, l_bounds, u_bounds)

    # Run optimization task multiple times
    pso.optimize(iterations=1000)

if __name__ == '__main__':
    main()
