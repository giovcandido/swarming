import logging
import numpy as np
import ray
import psutil

from tqdm import tqdm

from modules.Particle import Particle

class ParallelPSO:
    def __init__(self, swarm_size, dimension, function, lower_bounds, upper_bounds):
        self._w = 0.7
        self._c1 = 1.7
        self._c2 = 1.7

        self._swarm_size = swarm_size
        self._dimension = dimension

        # Get number of cpus available including logical threads
        num_cpus = psutil.cpu_count(logical=True)

        # Initialize ray instance
        ray.init(num_cpus=num_cpus, logging_level=logging.FATAL)

        # Save function as ray remote
        self._function = ray.remote(function)

        self._lower_bounds = lower_bounds
        self._upper_bounds = upper_bounds

    @property
    def particles(self):
        return self._particles
    
    @property
    def best_global_position(self):
        return self._best_global_position
    
    @property
    def best_global_score(self):
        return self._best_global_score
    
    def _initialize_search_space(self, swarm_size, dimension):
        self._particles = []

        # Initialize the particles in the swarm
        for _ in range(swarm_size):
            p = Particle(dimension, self._lower_bounds, self._upper_bounds)
            
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
                
    def optimize(self, max_iterations):
        # Move particles up to the maximum number of iterations
        for i in tqdm(range(max_iterations)):
            # If it's the first iteration, initialize the search space
            if i == 0:
                self._initialize_search_space(self._swarm_size, self._dimension)

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

    def _update_velocity(self, particle):
        inertia = self._w * particle.velocity
        
        r1 = np.random.uniform(0, 1)
        r2 = np.random.uniform(0, 1)
        
        cognitive_component = self._c1 * r1 * (particle.best_position - particle.position)
        social_component = self._c2 * r2 * (self._best_global_position - particle.position)

        particle.velocity = inertia + cognitive_component + social_component

    def _update_position(self, particle):
        particle.position = particle.position + particle.velocity

        self._clip_position(particle)

    def _clip_position(self, particle):
        particle.position = np.clip(particle.position, self._lower_bounds, self._upper_bounds)

    def _update_best_position(self, particle, score):
        if score < particle.best_score:
            particle.best_position = particle.position
            particle.best_score = score

            # Update the best global position, if necessary
            self._update_best_global_position(particle)

    def _update_best_global_position(self, particle):
        if particle.best_score < self._best_global_score:
            self._best_global_position = particle.best_position
            self._best_global_score = particle.best_score
