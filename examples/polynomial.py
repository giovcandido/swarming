from swarming.utils.argument_parser import parse_arguments

from swarming.core.pso import PSO, ParallelPSO

# Function to be optimized
def polynomial(x):
    return x[0] ** 2 + ((x[1] ** 2) / 16 - 5) ** 2 + 2 * x[0] + 6

def main():
    # Parse arguments, such as parallel, swarm size and more
    args = parse_arguments()

    parallel = args.parallel
    swarm_size = args.swarm_size
    iterations = args.iterations
    executions = args.executions

    # Function dimension
    dimension = 2

    # Now, let's set the bounds in order to restric the search space
    l_bounds = [-100, -100]
    u_bounds = [100, 100]

    # Select the desired PSO class
    if not parallel:
        PSOClass = PSO
    else:
        PSOClass = ParallelPSO

    # Create a PSO instance
    pso = PSOClass(swarm_size, dimension, polynomial, l_bounds, u_bounds)

    # Run optimization task multiple times
    pso.optimize(iterations, executions)

if __name__ == '__main__':
    main()
