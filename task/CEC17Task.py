from .AbstractTask import AbstractTask
from .AbstractFunc import AbstractFunc
import numpy as np

class CEC17Task(AbstractTask):
    def __init__(self, func: AbstractFunc, bound):
        self.func = func
        self.bound = bound

    def fitness(self, p):
        # for x in self.bound_decode(p, self.bound, dim=50):
        #     print(self.func(x))
        return np.array([self.func(x) for x in self.bound_decode(p, self.bound, dim=50)])    

def get_10_tasks():
    pass