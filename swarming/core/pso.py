import ray
import numpy as np

from logging import FATAL
from psutil import cpu_count
from time import time, strftime
from tqdm import tqdm

from swarming.utils.logger import Logger

from swarming.core.particle import Particle

# Get logger instance
logger = Logger.get_logger(__name__)

class PSO:

    def __init__(self, swarm_size, dimension, function, l_bounds, u_bounds):
        logger.info('Creating an instance of the PSO class')

        # Set inertia constant
        self._w = 0.7

        # Set cognitive constant
        self._c1 = 1.7

        # Set social constant
        self._c2 = 1.7

        logger.debug(f'Inertia Constant = {self._w}')
        logger.debug(f'Cognitive Constant = {self._c1}')
        logger.debug(f'Social Constant = {self._c2}')

        # Set number of particles in the swarm
        self._swarm_size = swarm_size

        # Set search space dimension
        self._dimension = dimension

        logger.debug(f'Swarm size = {self._swarm_size}')
        logger.debug(f'Dimension = {self._dimension}')

        # Set function to be optimized
        self._function = function

        logger.debug(f'Function = {self._function.__name__}')

        # Set search space boundaries
        self._lower_bounds = l_bounds
        self._upper_bounds = u_bounds

        logger.debug(f'Lower Bounds = {self._lower_bounds}')
        logger.debug(f'Upper Bounds = {self._upper_bounds}')

        logger.info('PSO instance was created successfully')

    def optimize(self, iterations, executions=1):
        logger.info(f'Task has {executions} executions')
        logger.info(f'With {iterations} iterations each execution')

        # Create lists to save output from multiple executions
        positions, scores = [], []

        # Run the algorithm many times
        # It helps to check if the restricted search space is appropriate
        for i in range(executions):
            # Set a random seed to achieve constant results
            np.random.seed(i)

            logger.debug(f'Random Seed = {i}')

            logger.info(f'Execution {i + 1} is in progress')

            start_time = time()

            # Get best position and best score from the current execution
            curr_position, curr_score = self._execute(iterations)

            end_time = time()

            logger.info(f'Execution {i + 1} ended successfully')

            logger.info(f'It took {end_time - start_time} s')

            # Save best position from the current execution
            positions.append(curr_position)

            # Save best score from the current execution
            scores.append(curr_score)

            logger.info(f'Execution best position = {curr_position}')
            logger.info(f'Execution Best score = {curr_score}')

        best_score_index = np.argmin(scores)

        logger.info(f'Execution {best_score_index + 1} had the best score')

        logger.info(f'Best task position = '
                    f'{positions[best_score_index]}')

        logger.info(f'Best task score = '
                    f'{scores[best_score_index]}')

        # Save task best position
        output_filename = 'task_best_position-' + strftime('%Y%m%d-%H%M%S')
        np.save(output_filename, positions[best_score_index])

    def _execute(self, iterations):
        logger.info(f'Creating the search space '
                    f'with {self._swarm_size} particles')

        # Initialize the search space
        self._initialize_search_space()

        logger.info('Search space was initialized successfully')

        logger.info('Moving particles around the search space')

        # Move particles up to the maximum number of iterations
        for i in tqdm(range(iterations)):
            logger.write(f'Iteration {i + 1}/{iterations}')

            # Loop over all particles in the swarm
            for particle in self._swarm:
                # Update velocity and position
                self._update(particle)

                # Calculate particle score
                score = self._function(particle.position)

                # If necessary, update the best position of the particle
                if score < particle.best_score:
                    self._update_best_position(particle, score)

            logger.write(f'Best iteration position = '
                         f'{self._best_swarm_position}')

            logger.write(f'Best iteration score = '
                         f'{self._best_swarm_score}')

        # Return the best swarm position as an approximate solution
        return self._best_swarm_position, self._best_swarm_score

    def _initialize_search_space(self):
        self._swarm = []

        # Initialize the particles in the swarm
        for i in tqdm(range(self._swarm_size)):
            # Create an instance of the particle class
            particle = Particle(self._dimension,
                                self._lower_bounds, self._upper_bounds)

            # Calculate particle score
            particle.best_score = self._function(particle.best_position)

            # Initialize best swarm position
            if i == 0:
                self._best_swarm_position = particle.best_position
                self._best_swarm_score = particle.best_score

            # Update best swarm position, if necessary
            if particle.best_score < self._best_swarm_score:
                self._update_best_swarm_position(particle)

            # Add particle to particles list
            self._swarm.append(particle)

    def _update(self, particle):
        # Generate two random numbers in [0, 1]
        r1 = np.random.uniform(0, 1)
        r2 = np.random.uniform(0, 1)

        # Calculate current velocity influence
        inertia_factor = self._w * particle.velocity

        # Calculate particle best position influence
        cognitive_factor = (self._c1
                            * r1
                            * (particle.best_position - particle.position))

        # Calculate swarm best position influence
        social_factor = (self._c2
                         * r2
                         * (self._best_swarm_position - particle.position))

        # Add all three factors to get the new velocity
        particle.velocity = inertia_factor + cognitive_factor + social_factor

        # Add current position and velocity to get the new position
        particle.position = particle.position + particle.velocity

        # Clip the new position, so that the particle stays within the bounds
        particle.position = np.clip(particle.position,
                                    self._lower_bounds, self._upper_bounds)

    def _update_best_position(self, particle, score):
        particle.best_position = particle.position
        particle.best_score = score

        # Update the swarm best position, if necessary
        if particle.best_score < self._best_swarm_score:
            self._update_best_swarm_position(particle)

    def _update_best_swarm_position(self, particle):
        self._best_swarm_position = particle.best_position
        self._best_swarm_score = particle.best_score


class ParallelPSO(PSO):

    def __init__(self, swarm_size, dimension, function, l_bounds, u_bounds):
        PSO.__init__(self, swarm_size, dimension, function, l_bounds, u_bounds)

        logger.info('Specializing instance as one of the ParallelPSO class')

        # Get number of cpus available including logical threads
        num_cpus = cpu_count(logical=True)

        logger.debug(f'Threads = {num_cpus}')

        # Initialize ray instance
        ray.init(num_cpus=num_cpus, logging_level=FATAL)

        logger.info('Ray was initialized successfully')

        # Save function as ray remote
        self._function = ray.remote(function)

        logger.debug(f'Function = {self._function}')

        logger.info('ParallelPSO instance was created successfully')

    def _execute(self, iterations):
        logger.info(f'Creating the search space '
                    f'with {self._swarm_size} particles')

        # Initialize the search space
        self._initialize_search_space()

        logger.info('Search space was initialized successfully')

        logger.info(f'Moving particles for {iterations} iterations')

        # Move particles up to the maximum number of iterations
        for i in tqdm(range(iterations)):
            logger.write(f'Iteration {i + 1}/{iterations}')

            # Loop over all particles in the swarm
            for particle in self._swarm:
                # Update velocity and position
                self._update(particle)

            scores = ray.get(
                [self._function.remote(p.position) for p in self._swarm])

            # Loop over all particles in the swarm
            for i, particle in enumerate(self._swarm):
                # If necessary, update the best position of the particle
                if scores[i] < particle.best_score:
                    self._update_best_position(particle, scores[i])

            logger.write(f'Best iteration position = '
                         f'{self._best_swarm_position}')

            logger.write(f'Best iteration score = '
                         f'{self._best_swarm_score}')

        # Return the best swarm position as an approximate solution
        return self._best_swarm_position, self._best_swarm_score

    def _initialize_search_space(self):
        self._swarm = []

        # Initialize the particles in the swarm
        for i in range(self._swarm_size):
            particle = Particle(self._dimension,
                                self._lower_bounds, self._upper_bounds)

            self._swarm.append(particle)

        scores = ray.get(
            [self._function.remote(p.best_position) for p in tqdm(self._swarm)])

        # Set best score of the firt particle
        self._swarm[0].best_score = scores[0]

        # Initialize best swarm position
        self._best_swarm_position = self._swarm[0].best_position
        self._best_swarm_score = self._swarm[0].best_score

        # Initialize the best score of the remaining particles
        for i, particle in enumerate(self._swarm[1:], 1):
            particle.best_score = scores[i]

            # Update best swarm position, if necessary
            if particle.best_score < self._best_swarm_score:
                self._update_best_swarm_position(particle)
