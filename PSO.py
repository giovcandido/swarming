import numpy as np

from Particle import Particle

class PSO:
    def __init__(self, swarm_size, dimension, function):
        self._w = 0.7
        self._c1 = 1.7
        self._c2 = 1.7

        self._function = function

        self._initialize_search_space(swarm_size, dimension)

    def _initialize_search_space(self, swarm_size, dimension):
        self._particles = []

        # Initialize the particles in the swarm
        for _ in range(swarm_size):
            p = Particle(dimension)

            self._particles.append(p)
        
        self._best_global_position = self._particles[0].best_position

        # Initialize the best position of the whole swarm
        for i in range(1, swarm_size):
            best_fit = self._function(self._particles[i].best_position)
            best_global_fit = self._function(self._best_global_position)

            if best_fit < best_global_fit:
                self._best_global_position = self._particles[i].best_position

    @property
    def particles(self):
        return self._particles
    
    @property
    def best_global_position(self):
        return self._best_global_position
    
    def optimize(self, max_iterations):
        # Move particles up to the maximum number of iterations
        for _ in range(max_iterations):
            # Loop over all particles in the swarm
            for particle in self._particles:
                # Update particle current velocity
                self._update_velocity(particle)

                # Move particle considering its new velocity
                self._update_position(particle)

                # If necessary, update the best position of the particle
                self._update_best_position(particle)

                # If necessary, update the best global position
                self._update_best_global_position(particle)
        
        # Return the best global position as an approximate solution
        return self._best_global_position

    def _update_velocity(self, particle):
        inertia = self._w * particle.velocity
        
        r1 = np.random.uniform(0, 1)
        r2 = np.random.uniform(0, 1)
        
        cognitive_component = self._c1 * r1 * (particle.best_position - particle.position)
        social_component = self._c2 * r2 * (self._best_global_position - particle.position)

        particle.velocity = inertia + cognitive_component + social_component

    def _update_position(self, particle):
        particle.position = particle.position + particle.velocity
    
    def _update_best_position(self, particle):
        fit = self._function(particle.position) 
        best_fit = self._function(particle.best_position)

        if fit < best_fit:
            particle.best_position = particle.position

    def _update_best_global_position(self, particle):
        best_fit = self._function(particle.best_position)
        best_global_fit = self._function(self._best_global_position)

        if best_fit < best_global_fit:
            self._best_global_position = particle.best_position