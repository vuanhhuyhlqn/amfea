import numpy as np
from mutation import *
from crossover import *
from rmp import *
from task import *
from typing import List
import time

class AMFEA:
    def __init__(self,
                 num_indis_per_task,
                 indi_len,
                 tasks : List[AbstractTask],
                 crossover: AbstractCrossover,
                 mutation: AbstractMutation,
                 rmp: AbstractRMP
                 ):
        self.indi_len = indi_len
        self.tasks = tasks
        self.num_tasks = len(tasks)
        self.pop_size = num_indis_per_task * self.num_tasks
        self.crossover = crossover
        self.mutation = mutation
        self.rmp = rmp
        self.pop = np.random.uniform(size=(self.pop_size, self.indi_len))
        self.skill_factor = np.zeros(self.pop_size, dtype=int)

        for i in range(self.pop_size):
            self.skill_factor[i] = i % self.num_tasks

        self.fitness = np.zeros(self.pop_size)
        self.best_fitness = np.zeros(self.num_tasks)
        self.mean_fitness = np.zeros(self.num_tasks)

        for task_id in range(self.num_tasks):
            task_mask = self.skill_factor == task_id
            self.fitness[task_mask] = tasks[task_id].fitness(self.pop[task_mask])
            self.best_fitness[task_id] = np.min(self.fitness[task_mask])
            self.mean_fitness[task_id] = np.mean(self.fitness[task_mask])

        print("Initialization:")
        for task_id in range(self.num_tasks):
            print("Task {0}:".format(task_id))
            print("Best Fitness: {0}".format(self.best_fitness[task_id]))
            print("Mean Fitness: {0}\n".format(self.mean_fitness[task_id]))

    def get_random_parents(self, num_pair):
        p1_indices = np.random.randint(0, self.pop_size, size=num_pair)
        p2_indices = np.random.randint(0, self.pop_size, size=num_pair)
        
        p1_skill_factor = self.skill_factor[p1_indices]
        p2_skill_factor = self.skill_factor[p2_indices]
        
        p1_fitness = self.fitness[p1_indices]
        p2_fitness = self.fitness[p2_indices]
        return self.pop[p1_indices], self.pop[p2_indices], p1_skill_factor, p2_skill_factor, p1_fitness, p2_fitness

    def get_random_individuals(self, num):
        p_indices = np.random.randint(0, self.pop_size, num)
        p = self.pop[p_indices]
        p_skill_factor = self.skill_factor[p_indices]
        p_fitness = self.fitness[p_indices]
        return p, p_skill_factor, p_fitness

    def evolve(self, gen, llm_rate):
        num_pair = np.random.randint(0, int(self.pop_size / 3))
        p1, p2, p1_skill_factor, p2_skill_factor, p1_fitness, p2_fitness = self.get_random_parents(num_pair)
        #Adaptive RMP

        armp = self.rmp(p1, p2, p1_skill_factor, p2_skill_factor, p1_fitness, p2_fitness, gen, llm_rate, self.tasks)
        
        #Crossover
        off, off_skill_factor = self.crossover(armp, p1, p2, p1_skill_factor, p2_skill_factor)
        off_fitness = np.zeros(len(off), dtype=np.float32)

        #Calculate crossover offsprings fitness
        for task_id in range(self.num_tasks):
            task_mask = off_skill_factor == task_id 
            off_fitness[task_mask] = self.tasks[task_id].fitness(off[task_mask])

        #Mutation
        num_mutation = np.random.randint(0, int(self.pop_size) / 3)
        off_mut, off_mut_skill_factor, off_mut_fitness = self.get_random_individuals(num_mutation)
        off_mut, off_mut_skill_factor = self.mutation(off_mut, off_mut_skill_factor)
        
        #Calculate mutation offsprings fitness
        for task_id in range(self.num_tasks):
            task_mask = off_mut_skill_factor == task_id 
            off_mut_fitness[task_mask] = self.tasks[task_id].fitness(off_mut[task_mask])
        
        ipop = np.concatenate([self.pop, off, off_mut])
        iskill_factor = np.concatenate([self.skill_factor, off_skill_factor, off_mut_skill_factor])
        ifitness = np.concatenate([self.fitness, off_fitness, off_mut_fitness])

        # print(ipop)
        # print(ifitness)

        self.pop = np.array([]).reshape(0, self.indi_len)
        self.fitness = []
        self.skill_factor = []

        for task_id in range(self.num_tasks):
            survive_size = int(self.pop_size / self.num_tasks)
            task_mask = iskill_factor == task_id
            
            tpop = ipop[task_mask]
            tfitness = ifitness[task_mask]
            assert(len(tpop) == len(tfitness))

            survive_indices = np.argpartition(tfitness, survive_size - 1)[:survive_size]
            
            self.best_fitness[task_id] = np.min(tfitness[survive_indices])
            self.mean_fitness[task_id] = np.mean(tfitness[survive_indices])

            self.pop = np.concatenate([self.pop, tpop[survive_indices]])
            self.fitness = np.concatenate([self.fitness, tfitness[survive_indices]])
            self.skill_factor = np.concatenate([self.skill_factor, np.full(survive_size, task_id)])
        
    def fit(self, num_eval=1, num_gen=1, monitor=False, monitor_rate=10, llm_rate=1000):
        #History Data
        bfs = np.zeros(shape=(self.num_tasks, num_gen + 1))
        mfs = np.zeros(shape=(self.num_tasks, num_gen + 1))

        for gen in range(num_gen + 1):
            start_time = time.time()
            self.evolve(gen, llm_rate)
            end_time = time.time()
            for task_id in range(self.num_tasks):
                bfs[task_id][gen] = self.best_fitness[task_id]
                mfs[task_id][gen] = self.mean_fitness[task_id]

            if gen % monitor_rate == 0:
                print("Gen {0}".format(gen))
                if monitor:
                    for task_id in range(self.num_tasks):
                        print("Task {0}, Best: {1}, Avg: {2}".format(task_id, self.best_fitness[task_id], self.mean_fitness[task_id]))
                print("Time taken each gen: %.4f seconds\n" % (end_time - start_time))
        return bfs, mfs
