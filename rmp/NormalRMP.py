import numpy as np
from .AbstractRMP import AbstractRMP

class NormalRMP(AbstractRMP):
    def __init__(self, default_rmp_value=0.3):
        self.default_rmp_value = default_rmp_value
        pass
    def get_rmp(self, size):
        return np.full(shape=(size, size), fill_value=0.3)
    def __call__(self, collect_state, p1, p2, p1_skill_factor, p2_skill_factor, p1_fitness, p2_fitness, gen, llm_rate, tasks):
        return self.get_rmp(len(tasks))