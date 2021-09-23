import numpy as np

from tqdm import tqdm

from utils.logger import Logger
from core.particle import Particle

# Get logger instance
logger = Logger.get_logger(__name__)

class PSO:

    def __init__(self, swarm_size, dimension, function, lower_bounds, upper_bounds):
        # Set inertia constant
        self._w = 0.7

        # Set cognitive constant
        self._c1 = 1.7

        # Set social constant
        self._c2 = 1.7

        self._swarm_size = swarm_size
        self._dimension = dimension

        self._function = function

        self._lower_bounds = lower_bounds
        self._upper_bounds = upper_bounds

    @property
    def particles(self):
        return self._particles
    
    @property
    def best_swarm_position(self):
        return self._best_swarm_position
    
    @property
    def best_swarm_score(self):
        return self._best_swarm_score
    
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

            # TODO: Show output from the current execution
        
        best_score_index = np.argmin(scores)
        
        # TODO: Show best output
        
        return positions[best_score_index], scores[best_score_index]

    def _run_task(self, iterations):
        # Move particles up to the maximum number of iterations
        for i in tqdm(range(iterations)):
            # If first iteration, initialize the search space
            if i == 0:
                self._initialize_search_space()

            # Loop over all particles in the swarm
            for particle in self._particles:
                # Update particle current velocity
                self._update_velocity(particle)

                # Move particle considering its new velocity
                self._update_position(particle)

                # Calculate particle score
                score = self._function(particle.position)

                # If necessary, update the best position of the particle
                self._update_best_position(particle, score)

        # Return the best swarm position as an approximate solution
        return self._best_swarm_position, self._best_swarm_score

    def _initialize_search_space(self):
        self._particles = []

        # Initialize the particles in the swarm
        for _ in range(self._swarm_size):
            p = Particle(self._dimension, self._lower_bounds, self._upper_bounds)

            self._particles.append(p)

        # Initialize the best position of the whole swarm
        for i, particle in enumerate(self._particles):
            particle.best_score = self._function(particle.best_position)

            if i == 0:
                self._best_swarm_position = particle.best_position
                self._best_swarm_score = particle.best_score

            if particle.best_score < self._best_swarm_score:
                self._best_swarm_position = particle.best_position
                self._best_swarm_score = particle.best_score

    def _update_velocity(self, particle):
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

    def _update_position(self, particle):
        # Add current position and velocity to get the new position
        particle.position = particle.position + particle.velocity

        # Clip the new position, so that the particle stays within the search space bounds
        particle.position = np.clip(particle.position, self._lower_bounds, self._upper_bounds)

    def _update_best_position(self, particle, score):
        # Update the particle best score, if necessary
        if score < particle.best_score:
            particle.best_position = particle.position
            particle.best_score = score

            # Update the swarm best position, if necessary
            if particle.best_score < self._best_swarm_score:
                self._best_swarm_position = particle.best_position
                self._best_swarm_score = particle.best_score
