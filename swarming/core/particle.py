from numpy import random, zeros

class Particle:

    def __init__(self, dimension, lower_bouds, upper_bounds):
        self._position = random.uniform(lower_bouds, upper_bounds, dimension)
        self._velocity = zeros(dimension)

        self._best_position = self._position

        self._best_score = None

    @property
    def position(self):
        return self._position

    @property
    def velocity(self):
        return self._velocity

    @property
    def best_position(self):
        return self._best_position

    @property
    def best_score(self):
        return self._best_score

    @position.setter
    def position(self, position):
        self._position = position

    @velocity.setter
    def velocity(self, velocity):
        self._velocity = velocity

    @best_position.setter
    def best_position(self, position):
        self._best_position = position

    @best_score.setter
    def best_score(self, score):
        self._best_score = score
