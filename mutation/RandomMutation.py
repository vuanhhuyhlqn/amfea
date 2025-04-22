import numpy as np
from .AbstractMutation import AbstractMutation

class RandomMutation(AbstractMutation):
    def __init__(self, bound, mutation_rate):
        self.bound = bound
        self.mutation_rate = mutation_rate
    
    def mutation(self, p, p_skill_factor):
        rnd = np.random.uniform(size=p.shape)
        new_values = np.random.uniform(-self.bound, self.bound, size=p.shape)
        replace_mask = rnd < self.mutation_rate
        off = p.copy()
        off_skill_factor = p_skill_factor.copy()
        off[replace_mask] = new_values[replace_mask]
        return off, off_skill_factor

    def __call__(self, p, p_skill_factor):
        return self.mutation(p, p_skill_factor)