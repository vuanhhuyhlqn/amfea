import numpy as np
from numba import jit

class AbstractTask:
    def __init__(self):
        pass
    def fitness(self, p):
        pass