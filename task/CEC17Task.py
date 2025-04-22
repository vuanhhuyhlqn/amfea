import numpy as np
from .AbstractTask import AbstractTask
from .cec2017.functions import all_functions

class CEC17Task(AbstractTask):
    def __init__(self, id):
        self.id = id
    def fitness(self, p):
        return all_functions[self.id - 1](p)
