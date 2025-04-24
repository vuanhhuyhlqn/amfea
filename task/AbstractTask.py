import numpy as np
from numba import jit

class AbstractTask:
    def __init__(self):
        pass
    def bound_decode(self, p, bound, dim):
        p_decoded = p[:, :dim] * (2 * bound) + (-bound)
        return p_decoded
    def fitness(self, p):
        pass