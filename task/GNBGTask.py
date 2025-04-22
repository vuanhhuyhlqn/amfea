import numpy as np
from .AbstractTask import AbstractTask
from .GNBG import GNBG_instances

class GNBGTask:
    def __init__(self, id):
        self.gnbg = GNBG_instances.get_gnbg(id)

    def fitness(self, p):
        return self.gnbg.fitness(p)