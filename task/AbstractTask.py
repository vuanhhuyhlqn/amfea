import numpy as np
from numby import jit

class AbstractTask:
    def __init__(self):
        self.evaluation_cnt = 0
    def bound_decode(self, p, bound, dim):
        # print(p)
        p_decoded = p[:, :dim] * (2 * bound) + (-bound)
        # print(p_decoded)
        return p_decoded
    def fitness(self, p):
        pass