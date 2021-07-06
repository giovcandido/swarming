# pso-optimization
Particle Swarm Optimization algorithm implemented with Python

## Instructions
In the optimization.py, change the function to be optimized to any function you wish.

Then, all you have to do is to execute the script. 

The script takes three arguments: (S) number of particles in the swarm, (M) maximum number of iterations and (T) number of times to run the algorithm.

Run as follows: python3 optimization.py --swarm-size S --max-iterations M --times T.
You can also run like this: python3 optimization.py -s S -m M -t T.

If you run the script with the example function and the restricted search space, considering s=20 particles and m=1000 iterations, the expected output has two possibilities. The first one is: x[0] = -1.0000 and x[1] = 8.9443. And the second one is: x[0] = -1.0000 and x[1] = -8.9443. 

Making it easier for you, run: python3 optimization.py --swarm-size 20 --max-iterations 1000. Check it out!