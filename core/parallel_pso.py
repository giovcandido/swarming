import logging
import ray
import psutil

from tqdm import tqdm

from core.pso import PSO
from core.particle import Particle

class ParallelPSO(PSO):

    def __init__(self, swarm_size, dimension, function, lower_bounds, upper_bounds):
        PSO.__init__(self, swarm_size, dimension, function, lower_bounds, upper_bounds)

        # Get number of cpus available including logical threads
        num_cpus = psutil.cpu_count(logical=True)

        # Initialize ray instance
        ray.init(num_cpus=num_cpus, logging_level=logging.FATAL)

        # Save function as ray remote
        self._function = ray.remote(function)

    def _run_task(self, iterations):
        # Move particles up to the maximum number of iterations
        for i in tqdm(range(iterations)):
            # If it's the first iteration, initialize the search space
            if i == 0:
                self._initialize_search_space()

            # Loop over all particles in the swarm
            for particle in self._swarm:
                # Update velocity and position
                self._update(particle)

            scores = ray.get([self._function.remote(p.position) for p in self._swarm])

            # Loop over all particles in the swarm
            for j, particle in enumerate(self._swarm):
                # If necessary, update the best position of the particle
                if scores[j] < particle.best_score:
                    self._update_best_position(particle, scores[j])

        # Return the best swarm position as an approximate solution
        return self._best_swarm_position, self._best_swarm_score

    def _initialize_search_space(self):
        self._swarm = []

        # Initialize the particles in the swarm
        for _ in range(self._swarm_size):
            p = Particle(self._dimension, self._lower_bounds, self._upper_bounds)
            
            self._swarm.append(p)
            
        best_scores = ray.get([self._function.remote(p.best_position) for p in self._swarm])

        # Initialize the best position of the whole swarm
        for i, particle in enumerate(self._swarm):
            particle.best_score = best_scores[i]

            if i == 0:
                self._best_swarm_position = particle.best_position
                self._best_swarm_score = particle.best_score
            
            # Update best swarm position, if necessary
            if particle.best_score < self._best_swarm_score:
                self._update_best_swarm_position(particle)
