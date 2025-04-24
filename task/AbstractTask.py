import numpy as np
from numba import jit

class AbstractTask:
    def __init__(self):
        pass
    def decode(self, x, bound):
        return x * (2 * bound) + (-bound)
    def fitness(self, p):
        pass