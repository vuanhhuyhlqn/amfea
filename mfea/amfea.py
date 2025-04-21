import numpy as np
from crossover.crossover import AbstractCrossover, ArithmeticCrossover
from task.task import AbstractTask
from mutation.mutation import AbstractMutation, AdditiveMutation, RandomMutation
from rmp.rmp import AbstractRMP, NormalRMP
from typing import List

class AMFEA:
    def __init__(self,
                 pop_size,
                 indi_len,
                 bound,
                 tasks : List[AbstractTask],
                 crossover: AbstractCrossover,
                 mutation: AbstractMutation,
                 rmp: AbstractRMP
                 ):
        self.pop_size = pop_size
        self.indi_len = indi_len
        self.tasks = tasks
        self.bound = bound
        self.num_tasks = len(tasks)
        self.crossover = crossover
        self.mutation = mutation

        self.pop = np.random.uniform(-self.bound, self.bound, size=(self.pop_size, self.indi_len))
        self.skill_factor = np.zeros(pop_size, dtype=int)

        for i in range(self.pop_size):
            self.skill_factor[i] = i % self.num_tasks

        self.fitness = np.zeros(self.pop_size)

        for task_id in range(len(tasks)):
            task_mask = self.skill_factor == task_id
            self.fitness[task_mask] = tasks[task_id].fitness(self.pop[task_mask])

    def evolve(self):
        num_pair = np.random.randint(0, self.pop_size)
        p1_indices = np.random.randint(0, self.pop_size, size=num_pair)
        p2_indices = np.random.randint(0, self.pop_size, size=num_pair)
        
        p1_skill_factor = self.skill_factor[p1_indices]
        p2_skill_factor = self.skill_factor[p2_indices]
        
        p1_fitness = self.fitness[p1_indices]
        p2_fitness = self.fitness[p2_indices]




    def fit(self, num_gen, llm_rate):
        for gen in range(num_gen):
            pass
