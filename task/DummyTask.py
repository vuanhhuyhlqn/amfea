import numpy as np
from .AbstractTask import AbstractTask

class DummyTask(AbstractTask):
    def __init__(self):
        super().__init__()

    def fitness(self, p):
        return np.sum(p, axis=1)