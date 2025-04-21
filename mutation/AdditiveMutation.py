import numpy as np
from .AbstractMutation import AbstractMutation
    
class AdditiveMutation(AbstractMutation):
    def __init__(self, bound, delta):
        self.bound = bound
        self.delta = delta
    
    def mutation(self, p, p_skill_factor):
        off = p.copy()
        off_skill_factor = p_skill_factor.copy()
        deltas = np.random.uniform(-self.delta, self.delta, size=p.shape)
        off += deltas
        off[off > self.bound] = self.bound
        off[off < -self.bound] = -self.bound
        return off, off_skill_factor
    
    def __call__(self, p, p_skill_factor):
        return self.mutation(p, p_skill_factor)