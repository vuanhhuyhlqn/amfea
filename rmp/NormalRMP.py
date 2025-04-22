import numpy as np
from .AbstractRMP import AbstractRMP

class NormalRMP(AbstractRMP):
    def __init__(self, default_rmp_value=0.3):
        self.default_rmp_value = default_rmp_value
        pass
    def get_rmp(self, size):
        return np.full(size, self.default_rmp_value)
    def __call__(self, p1, p2, p1_skill_factor, p2_skill_factor, p1_fitness, p2_fitness):
        return self.get_rmp(len(p1))