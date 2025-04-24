from .AbstractTask import AbstractTask
from .AbstractFunc import AbstractFunc
from .CEC17Func import *
import numpy as np
from typing import Tuple, Type, List

class CEC17Task(AbstractTask):
    def __init__(self, func: AbstractFunc, bound):
        self.func = func
        self.bound = bound

    def fitness(self, p):
        # for x in self.bound_decode(p, self.bound, dim=50):
        #     print(self.func(x))
        return np.array([self.func(x) for x in self.bound_decode(p, self.bound, dim=50)])    

def get_10_tasks() -> List[AbstractFunc]:
    griewank_shift = np.zeros(50)
    griewank_shift[:25] = -80
    griewank_shift[25:] = 80
    rastrigin_shift = np.zeros(50)
    rastrigin_shift[:25] = 40
    rastrigin_shift[25:] = -40
    tasks = [
        CEC17Task(Sphere(50, np.full(50, 0)), 100),
        CEC17Task(Sphere(50, np.full(50, 80)), 100),
        CEC17Task(Sphere(50, np.full(50, -80)), 100),
        CEC17Task(Weierstrass(25, np.full(25, -0.4)), 0.5),
        CEC17Task(Rosenbrock(50, np.full(50, -1)), 50),
        CEC17Task(Ackley(50, np.full(50, 40)), 50),
        CEC17Task(Schwefel(50, np.full(50, 0)), 500),
        CEC17Task(Griewank(50, griewank_shift), 100),
        CEC17Task(Rastrigin(50, rastrigin_shift), 50),
    ]
    return tasks