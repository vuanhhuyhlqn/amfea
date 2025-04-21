import numpy as np
from .AbstractRMP import AbstractRMP

class NormalRMP(AbstractRMP):
    def __init__(self):
        pass
    def get_rmp(self, size):
        return np.full(size, 0.3)
    def __call__(self, p1, p2, p1_skill_factor, p2_skill_factor, p1_fitness, p2_fitness):
        return self.get_rmp(len(p1))