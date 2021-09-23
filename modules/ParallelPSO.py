import logging
import ray
import psutil

from tqdm import tqdm

from modules.PSO import PSO
from modules.Particle import Particle

class ParallelPSO(PSO):
    def __init__(self, swarm_size, dimension, function, lower_bounds, upper_bounds):
        PSO.__init__(self, swarm_size, dimension, function, lower_bounds, upper_bounds)

        # Get number of cpus available including logical threads
        num_cpus = psutil.cpu_count(logical=True)

        # Initialize ray instance
        ray.init(num_cpus=num_cpus, logging_level=logging.FATAL)

        # Save function as ray remote
        self._function = ray.remote(function)

    def optimize(self, iterations):
        # Move particles up to the maximum number of iterations
        for i in tqdm(range(iterations)):
            # If it's the first iteration, initialize the search space
            if i == 0:
                self._initialize_search_space()

            # Loop over all particles in the swarm
            for particle in self._particles:
                # Update particle current velocity
                self._update_velocity(particle)

                # Move particle considering its new velocity
                self._update_position(particle)

            scores = ray.get([self._function.remote(p.position) for p in self._particles])

            # Loop over all particles in the swarm
            for j, particle in enumerate(self._particles):
                # If necessary, update the best position of the particle
                self._update_best_position(particle, scores[j])

        # Return the best global position as an approximate solution
        return self._best_global_position, self._best_global_score

    def _initialize_search_space(self):
        self._particles = []

        # Initialize the particles in the swarm
        for _ in range(self._swarm_size):
            p = Particle(self._dimension, self._lower_bounds, self._upper_bounds)
            
            self._particles.append(p)
            
        best_scores = ray.get([self._function.remote(p.best_position) for p in self._particles])

        # Initialize the best position of the whole swarm
        for i, particle in enumerate(self._particles):
            particle.best_score = best_scores[i]

            if i == 0:
                self._best_global_position = particle.best_position
                self._best_global_score = particle.best_score
            
            if particle.best_score < self._best_global_score:
                self._best_global_position = particle.best_position
                self._best_global_score = particle.best_score
