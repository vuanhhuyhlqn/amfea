import numpy as np
from .AbstractTask import AbstractTask
from .cec2017.functions import all_functions
from .cec2017.basic import griewank, rastrigin, rosenbrock, ackley

class CEC17Task(AbstractTask):
    def __init__(self, id):
        self.id = id
    def fitness(self, p):
        return all_functions[self.id - 1](p)

class GriewankTask(AbstractTask):
    def __init__(self):
        pass
    def fitness(self, p):
        return griewank(p)
    
class RastriginTask(AbstractTask):
    def __init__(self):
        pass
    def fitness(self, p):
        return rastrigin(p)

class RosenbrockTask(AbstractTask):
    def __init__(self):
        pass
    def fitness(self, p):
        return rosenbrock(p)

class AckleyTask(AbstractTask):
    def __init__(self):
        pass
    def fitness(self, p):
        return ackley(p)
