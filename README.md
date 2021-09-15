# PSO Optimization

Particle Swarm Optimization algorithm with serial and distributed execution. 

It's made with Python3 and tested on Linux.

## Installation

Clone the repository or download the compressed source code. If you opted for the latter, you need to extract the source code to a desired directory.

In both cases, open the project directory in your terminal. 

Now, run the following command:
```bash
sh install_deps.sh
```

In case you can't install the dependencies as a user, run the following instead:
```bash
sudo sh install_deps.sh
```

## Usage

The optimization script takes four arguments: 
- (P) if the algorithm should run in parallel;
- (S) the number of particles in the swarm;
- (M) the maximum number of iterations;
- (T) the number of times to run the algorithm.

You can run it as follows:
```bash
python3 optimization.py --parallel P --swarm-size S --max-iterations M --times T
``` 

You can also run: 
```bash
python3 optimization.py -p P -s S -m M -t T
``` 

Note that you should __replace__ the uppercase letters with the values you wish.

Now that you have all the requirements installed and know how to execute the script, open the optimization.py file. In this file, you can change the function to be optimized. However, after changing the function, you should also update the dimension variable and the lower and upper bounds to each of these variables.

Before making your changes to the script, you may want to test it using the predefined values. In this case, check the example section below.

## Example

In order to test it for the first time, keep the script unchanged and run:
```bash
python3 optimization.py --parallel n --swarm-size 20 --max-iterations 1000 --times 1
```

If you want to test the distributed execution, change the first argument to 'y':
```bash
python3 optimization.py --parallel y --swarm-size 20 --max-iterations 1000 --times 1
```

Running any of the above commands with the optimization script unchanged, you should expect one of two possible outputs.

The first one is: 
- x[0] = -1.0000;
- x[1] = 8.9443. 

And the second one is: 
- x[0] = -1.0000;
- x[1] = -8.9443. 

__Bear in mind__ that distributed computing works better for long calculations.

## Acknowledgements

The idea of Particle Swarm Optimization can be found in [Kennedy's paper](https://ieeexplore.ieee.org/document/488968).

It's worth mentioning that my colleague [gugarosa](https://github.com/gugarosa) made sure I was on the right path.
