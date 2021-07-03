import argparse

from PSO import PSO

parser = argparse.ArgumentParser()

parser.add_argument('-s', '--swarm-size', help="number of particles in the swarm", required=True)
parser.add_argument('-m', '--max-iterations', help="maximum number of iterations", required=True)

# Parse swarm size and maximum number of iterations
args = parser.parse_args()

# Set the number of particles in the swarm
swarm_size = int(args.swarm_size)

# Set the dimension of the problem
dimension = 2

# And the function to be optimized
function = lambda x: x[0] ** 2 + ((x[1] ** 2) / 16 - 5) ** 2 + 2 * x[0] + 6

# Create a PSO instance
pso = PSO(swarm_size, dimension, function)

# Set maximum number of iterations
max_iterations = int(args.max_iterations)

# Find an approximate solution with PSO
approx_sol = pso.optimize(max_iterations)

print('\nThe approximate solution is:')

for i, x in enumerate(approx_sol):
    x = round(x, 4) + 0
    print('x[%i] = %.4f' % (i, x))

print('')