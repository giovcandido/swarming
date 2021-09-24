import numpy as np

from tqdm import tqdm

from core.particle import Particle

class PSO:

    def __init__(self, swarm_size, dimension, function, lower_bounds, upper_bounds):
        # Set inertia constant
        self._w = 0.7

        # Set cognitive constant
        self._c1 = 1.7

        # Set social constant
        self._c2 = 1.7

        # Set number of particles in the swarm
        self._swarm_size = swarm_size
        
        # Set search space dimension
        self._dimension = dimension

        # Set function to be optimized
        self._function = function

        # Set search space boundaries
        self._lower_bounds = lower_bounds
        self._upper_bounds = upper_bounds

    def optimize(self, iterations, executions):
        # Create lists to save output from multiple executions
        positions, scores = [], []

        # Run the algorithm many times
        # It helps to check if the restricted search space is appropriate
        for i in range(executions):
            # Set a random seed to achieve constant results
            np.random.seed(i)
            
            # Get best position and best score from the current execution
            curr_position, curr_score = self._run_task(iterations)

            # Save best position from the current execution
            positions.append(curr_position)

            # Save best score from the current execution
            scores.append(curr_score)
        
        best_score_index = np.argmin(scores)

        return positions[best_score_index], scores[best_score_index]

    def _run_task(self, iterations):
        # Initialize the search space
        self._initialize_search_space()

        # Move particles up to the maximum number of iterations
        for _ in tqdm(range(iterations)):
            # Loop over all particles in the swarm
            for particle in self._swarm:
                # Update velocity and position
                self._update(particle)
                
                # Calculate particle score
                score = self._function(particle.position)

                # If necessary, update the best position of the particle
                if score < particle.best_score:
                    self._update_best_position(particle, score)

        # Return the best swarm position as an approximate solution
        return self._best_swarm_position, self._best_swarm_score

    def _initialize_search_space(self):
        self._swarm = []

        # Initialize the particles in the swarm
        for i in tqdm(range(self._swarm_size)):
            # Create an instance of the particle class
            particle = Particle(self._dimension, self._lower_bounds, self._upper_bounds)

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
        cognitive_factor = self._c1 * r1 * (particle.best_position - particle.position)
        
        # Calculate swarm best position influence
        social_factor = self._c2 * r2 * (self._best_swarm_position - particle.position)

        # Add all three factors to get the new velocity
        particle.velocity = inertia_factor + cognitive_factor + social_factor

        # Add current position and velocity to get the new position
        particle.position = particle.position + particle.velocity

        # Clip the new position, so that the particle stays within the search space bounds
        particle.position = np.clip(particle.position, self._lower_bounds, self._upper_bounds)

    def _update_best_position(self, particle, score):
        particle.best_position = particle.position
        particle.best_score = score

        # Update the swarm best position, if necessary
        if particle.best_score < self._best_swarm_score:
            self._update_best_swarm_position(particle)
    
    def _update_best_swarm_position(self, particle):
        self._best_swarm_position = particle.best_position
        self._best_swarm_score = particle.best_score
