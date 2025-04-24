import numpy as np
from .AbstractTask import AbstractTask
from .GNBG import GNBG_instances

class GNBGTask(AbstractTask):
    def __init__(self, id, bound):
        self.gnbg = GNBG_instances.get_gnbg(id)
        self.bound = bound
    def fitness(self, p):
        return self.gnbg.fitness(self.bound_decode(p, self.bound, 50))