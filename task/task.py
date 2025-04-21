import numpy as np
from numba import jit

class AbstractTask:
    def __init__(self):
        pass
    def fitness(self, p):
        pass

class DummyTask(AbstractTask):
    def __init__(self):
        super().__init__()

    def fitness(self, p):
        return np.sum(p, axis=1)