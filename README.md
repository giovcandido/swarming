# pso-optimization
Particle Swarm Optimization algorithm with serial and distributed execution. Made with Python3. 

## How to use
The first thing you have to do is: 
* sh install_deps.sh

Then, in the optimization.py, change the function to be optimized if you wish.

Now, all you have to do is to execute the script. 

The script takes four arguments: (P) parallel execution or not, (S) number of particles in the swarm, (M) maximum number of iterations and (T) number of times to run the algorithm.

You can run as follows: python3 optimization.py --parallel P --swarm-size S --max-iterations M --times T. Or: python3 optimization.py -p P -s S -m M -t T. If you want to run the algorithm using Ray, execute with -p y. In case you want to execute serially, use -p n. 

If you run the script with the example function and the restricted search space, considering s=20 particles and m=1000 iterations, the expected output has two possibilities. The first one is: x[0] = -1.0000 and x[1] = 8.9443. And the second one is: x[0] = -1.0000 and x[1] = -8.9443. 

Making it easier for you, run:
* python3 optimization.py --parallel n --swarm-size 20 --max-iterations 1000 --times 1. 
 
See it for yourself.

Bear in mind that distributed computing works better for long calculations.

# Acknowledgements
The ideia of Particle Swarm Optimization can be found in [Kennedy's paper](https://ieeexplore.ieee.org/document/488968).

It's worth mentioning that my colleague [gugarosa](https://github.com/gugarosa) made sure I was on the right path.
