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