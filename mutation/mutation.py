import numpy as np

class AbstractMutation:
    def __init__(self):
        pass
    def __call__(self, p):
        pass

class RandomMutation(AbstractMutation):
    def __init__(self, bound):
        self.bound = bound
    
    def mutation(self, p):
        rnd = np.random.uniform(size=p.shape)
        new_values = np.random.uniform(-self.bound, self.bound, size=p.shape)
        replace_mask = rnd > 0.5
        off = p.copy()
        off[replace_mask] = new_values[replace_mask]
        return off

    def __call__(self, p):
        return self.mutation(p)