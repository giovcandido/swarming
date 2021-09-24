import logging
import ray
import psutil

from tqdm import tqdm

from utils.logger import Logger

from core.pso import PSO
from core.particle import Particle

# Get logger instance
logger = Logger.get_logger(__name__)

class ParallelPSO(PSO):

    def __init__(self, swarm_size, dimension, function, lower_bounds, upper_bounds):
        PSO.__init__(self, swarm_size, dimension, function, lower_bounds, upper_bounds)

        logger.info('Specializing instance as one of the ParallelPSO class')

        # Get number of cpus available including logical threads
        num_cpus = psutil.cpu_count(logical=True)

        logger.debug('Threads = %s' % (num_cpus))

        # Initialize ray instance
        ray.init(num_cpus=num_cpus, logging_level=logging.FATAL)
        
        logger.info('Ray was initialized successfully')

        # Save function as ray remote
        self._function = ray.remote(function)

        logger.debug('Function = %s' % (self._function))

        logger.info('ParallelPSO instance was created successfully')

    def _execute(self, iterations):
        logger.info('Initializing the search space with %s particles' % (self._swarm_size))

        # Initialize the search space
        self._initialize_search_space()

        logger.info('Search space was initialized successfully')

        logger.info('Moving particles around the search space for %s iterations' % (iterations))

        # Move particles up to the maximum number of iterations
        for i in tqdm(range(iterations)):
            logger.write('Iteration %s/%s' % (i + 1, iterations))

            # Loop over all particles in the swarm
            for particle in self._swarm:
                # Update velocity and position
                self._update(particle)

            scores = ray.get([self._function.remote(p.position) for p in self._swarm])

            # Loop over all particles in the swarm
            for i, particle in enumerate(self._swarm):
                # If necessary, update the best position of the particle
                if scores[i] < particle.best_score:
                    self._update_best_position(particle, scores[i])

            logger.write('Iteration best position = %s' % (self._best_swarm_position))
            logger.write('Iteration best score = %s' % (self._best_swarm_score))
            
        # Return the best swarm position as an approximate solution
        return self._best_swarm_position, self._best_swarm_score

    def _initialize_search_space(self):
        self._swarm = []

        # Initialize the particles in the swarm
        for i in range(self._swarm_size):
            particle = Particle(self._dimension, self._lower_bounds, self._upper_bounds)

            self._swarm.append(particle)
            
        scores = ray.get([self._function.remote(p.best_position) for p in tqdm(self._swarm)])

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
