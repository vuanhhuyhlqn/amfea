import numpy as np
from .AbstractRMP import AbstractRMP

class NormalRMP(AbstractRMP):
    def __init__(self):
        pass
    def get_rmp(self, size):
        return np.full(size, 0.3)
    def __call__(self, size):
        return self.get_rmp(size)