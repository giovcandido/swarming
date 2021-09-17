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

An optimization script takes four arguments: 
- (P) if the algorithm should run in parallel;
- (S) the number of particles in the swarm;
- (M) the maximum number of iterations;
- (T) the number of times to run the algorithm.

You can run it as follows:
```bash
python3 script.py --parallel P --swarm-size S --max-iterations M --times T
``` 

You can also run: 
```bash
python3 script.py -p P -s S -m M -t T
``` 

Note that you should __replace__ script.py with the desired script and the uppercase letters with the values you wish.

Moreover, it's also important to note that 'parallel' and 'times' are optional arguments. If you don't include them when executing a script, the default values are considered. For parallel, the default is 'n', which means 'no'. As for times, the default is one time.

Now that you have all the requirements installed and know how to execute a script, open the script you want to work with. In the file, you can change the function to be optimized. However, after changing the function, you should also update the function dimension and the lower and upper bounds to each of the function variables.

Before making your changes, you may want to test the using the predefined values. In this case, check the example section below.

## Example

There are currently two predefined optimization scripts:
- math_function.py;
- neural_net.py.

Let's pick the first one to use as an example.

In order to test it for the first time, keep the script unchanged and run:
```bash
python3 math_function.py --parallel n --swarm-size 20 --max-iterations 1000 --times 1
```

If you want to test the distributed execution, change the first argument to 'y':
```bash
python3 math_function.py --parallel y --swarm-size 20 --max-iterations 1000 --times 1
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
