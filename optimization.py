import argparse

from PSO import PSO

parser = argparse.ArgumentParser()

parser.add_argument('-s', '--swarm-size', help="number of particles in the swarm", required=True)
parser.add_argument('-m', '--max-iterations', help="maximum number of iterations", required=True)
parser.add_argument('-t', '--times', help="number of times to run", required=True)

# Parse swarm size and maximum number of iterations
args = parser.parse_args()

# Set the number of particles in the swarm
swarm_size = int(args.swarm_size)

# Set the dimension of the problem
dimension = 2

# And the function to be optimized
function = lambda x: x[0] ** 2 + ((x[1] ** 2) / 16 - 5) ** 2 + 2 * x[0] + 6

# Now, let's set the bounds in order to restric the search space

# Lower bounds
lower_bounds = (-100, -100)

# Upper bounds
upper_bounds = (100, 100)

# Number of times to run the algorithm
times_to_run = int(args.times)

# Run the PSO algorithm many times
# It helps to check if the restricted search space is appropriate

print('Running algorithm %i times...\n' % times_to_run)

for i in range(1, times_to_run + 1):
    print('==> Run number %i' % i)

    # Create a PSO instance
    pso = PSO(swarm_size, dimension, function, lower_bounds, upper_bounds)

    # Set maximum number of iterations
    max_iterations = int(args.max_iterations)

    # Find an approximate solution with PSO
    approx_sol = pso.optimize(max_iterations)

    print('Approximate solution:')

    for i, x in enumerate(approx_sol):
        x = round(x, 4) + 0
        print('x[%i] = %.4f' % (i, x))

    print('')