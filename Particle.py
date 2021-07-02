import numpy as np

class Particle:
    def __init__(self, dimension):
        self._position = np.random.rand(dimension)
        self._velocity = np.random.rand(dimension)

        self._best_position = self._position

    @property
    def position(self):
        return self._position
    
    @property
    def velocity(self):
        return self._velocity
    
    @property
    def best_position(self):
        return self._best_position
    
    @position.setter
    def position(self, position):
        self._position = position
    
    @velocity.setter
    def velocity(self, velocity):
        self._velocity = velocity

    @best_position.setter
    def best_position(self, position):
        self._best_position = position