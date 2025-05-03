from .AbstractTask import AbstractTask
from .AbstractFunc import AbstractFunc
from .CEC17Func import *
import numpy as np
from scipy.io import loadmat
from typing import Tuple, Type, List
import os

path = os.path.dirname(os.path.realpath(__file__))

class CEC17Task(AbstractTask):
    def __init__(self, func: AbstractFunc, bound):
        self.func = func
        self.bound = bound

    def fitness(self, p):
        return np.array([self.func(x) for x in self.bound_decode(p, self.bound, dim=self.func.dim)])    

def get_10_tasks() -> List[AbstractTask]:
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
        CEC17Task(Weierstrass(50, np.full(50, -0.4)), 0.5),
        CEC17Task(Schwefel(50, np.full(50, 0)), 500),
        CEC17Task(Griewank(50, griewank_shift), 100),
        CEC17Task(Rastrigin(50, rastrigin_shift), 50),
    ]
    return tasks

def get_2_tasks(ID: int) -> List[AbstractTask]:
    tasks = []
    task_names = ["CI_H", "CI_M", "CI_L", "PI_H", "PI_M", "PI_L", "NI_H", "NI_M", "NI_L"]
    task_name = task_names[ID]
    print(task_name)
    if task_name == "CI_H":
        ci_h = loadmat(path + "/CEC17/Tasks/CI_H.mat")
        shift = ci_h['GO_Task1']
        rotation_matrix = ci_h['Rotation_Task1']
        tasks.append(CEC17Task(Griewank(50, shift, rotation_matrix), 100))
        shift = ci_h['GO_Task2']
        rotation_matrix = ci_h['Rotation_Task2']
        tasks.append(CEC17Task(Rastrigin(50, shift, rotation_matrix), 50))

    if task_name == "CI_M":
        ci_m = loadmat(path + "/CEC17/Tasks/CI_M.mat")
        shift = ci_m['GO_Task1']
        rotation_matrix = ci_m['Rotation_Task1']
        tasks.append(CEC17Task(Ackley(50, shift, rotation_matrix), 50))
        shift = ci_m['GO_Task2']
        rotation_matrix = ci_m['Rotation_Task2']
        tasks.append(CEC17Task(Rastrigin(50, shift, rotation_matrix), 50))
    
    if task_name == "CI_L":
        ci_l = loadmat(path + "/CEC17/Tasks/CI_L.mat")
        shift = ci_l['GO_Task1']
        rotation_matrix = ci_l['Rotation_Task1']
        tasks.append(CEC17Task(Ackley(50, shift, rotation_matrix), 100))
        # shift = ci_l['GO_Task2']
        # rotation_matrix = ci_l['Rotation_Task2']
        # tasks.append(CEC17Task(Schwefel(50, shift, rotation_matrix), 500))
        tasks.append(CEC17Task(Schwefel(50), 500))
    
    if task_name == "PI_H":
        pi_h = loadmat(path + "/CEC17/Tasks/PI_H.mat")
        shift = pi_h['GO_Task1']
        rotation_matrix = pi_h['Rotation_Task1']
        tasks.append(CEC17Task(Rastrigin(50, shift, rotation_matrix), 50))
        shift = pi_h['GO_Task2']
        # rotation_matrix = pi_h['Rotation_Task2']
        tasks.append(CEC17Task(Sphere(50, shift), 100))

    if task_name == "PI_M":
        pi_m = loadmat(path + "/CEC17/Tasks/PI_M.mat")
        shift = pi_m['GO_Task1']
        rotation_matrix = pi_m['Rotation_Task1']
        tasks.append(CEC17Task(Ackley(50, shift, rotation_matrix), 50))
        tasks.append(CEC17Task(Rosenbrock(50), 50))
    
    if task_name == "PI_L":
        pi_l = loadmat(path + "/CEC17/Tasks/PI_L.mat")
        shift = pi_l['GO_Task1']
        rotation_matrix = pi_l['Rotation_Task1']
        tasks.append(CEC17Task(Ackley(50, shift, rotation_matrix), 50))
        shift = pi_l['GO_Task2']
        rotation_matrix = pi_l['Rotation_Task2']
        tasks.append(CEC17Task(Weierstrass(25, shift, rotation_matrix), 0.5))
    
    if task_name == "NI_H":
        ni_h = loadmat(path + "/CEC17/Tasks/NI_H.mat")
        tasks.append(CEC17Task(Rosenbrock(50), 50))
        shift = ni_h['GO_Task2']
        rotation_matrix = ni_h['Rotation_Task2']
        tasks.append(CEC17Task(Rastrigin(50, shift, rotation_matrix), 50))

    if task_name == "NI_M":
        ni_m = loadmat(path + "/CEC17/Tasks/NI_M.mat")
        shift = ni_m['GO_Task1']
        rotation_matrix = ni_m['Rotation_Task1']
        tasks.append(CEC17Task(Griewank(50, shift, rotation_matrix), 100))
        shift = ni_m['GO_Task2']
        rotation_matrix = ni_m['Rotation_Task2']
        tasks.append(CEC17Task(Rastrigin(50, shift, rotation_matrix), 50))

    if task_name == "NI_L":
        ni_l = loadmat(path + "/CEC17/Tasks/NI_L.mat")
        shift = ni_l['GO_Task1']
        rotation_matrix = ni_l['Rotation_Task1']
        tasks.append(CEC17Task(Rastrigin(50, shift, rotation_matrix), 50))
        tasks.append(CEC17Task(Schwefel(50, shift, rotation_matrix), 500))
    
    return tasks, task_name