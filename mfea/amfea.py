import numpy as np
from crossover.crossover import AbstractCrossover, ArithmeticCrossover
from task.task import AbstractTask
from typing import List

class AMFEA:
    def __init__(self, pop_size, indi_len, bound, tasks : List[AbstractTask], crossover: AbstractCrossover):
        self.pop_size = pop_size
        self.indi_len = indi_len
        self.tasks = tasks
        self.bound = bound
        self.num_tasks = len(tasks)
        self.crossover = crossover

        self.pop = np.random.uniform(-self.bound, self.bound, size=(self.pop_size, self.indi_len))
        self.skill_factor = np.zeros(pop_size, dtype=int)

        for i in range(self.pop_size):
            self.skill_factor[i] = i % self.num_tasks

        self.fitness = np.zeros(self.pop_size)
        print(len(tasks))

        for task_id in range(len(tasks)):
            print(task_id)

            task_mask = self.skill_factor == task_id
            self.fitness[task_mask] = tasks[task_id].fitness(self.pop[task_mask])

    
