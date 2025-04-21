import numpy as np

class AbstractMutation:
    def __init__(self):
        pass
    def __call__(self, p):
        pass

class RandomMutation(AbstractMutation):
    def __init__(self, bound, mutation_rate):
        self.bound = bound
        self.mutation_rate = mutation_rate
    
    def mutation(self, p):
        rnd = np.random.uniform(size=p.shape)
        new_values = np.random.uniform(-self.bound, self.bound, size=p.shape)
        replace_mask = rnd < self.mutation_rate
        off = p.copy()
        off[replace_mask] = new_values[replace_mask]
        return off

    def __call__(self, p):
        return self.mutation(p)
    
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
    
    def __call__(self, p):
        return self.mutation(p)