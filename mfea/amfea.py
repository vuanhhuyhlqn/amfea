import numpy as np
from mutation import *
from crossover import *
from rmp import *
from task import *
from typing import List
import time
from scipy.stats import pearsonr

class AMFEA:
    def __init__(self,
                 num_indis_per_task,
                 indi_len,
                 tasks : List[AbstractTask],
                 crossover: AbstractCrossover,
                 mutation: AbstractMutation,
                 rmp: AbstractRMP,
                 ):
        self.indi_len = indi_len
        self.tasks = tasks
        self.num_tasks = len(tasks)
        self.pop_size = num_indis_per_task * self.num_tasks
        self.crossover = crossover
        self.mutation = mutation
        self.rmp = rmp
        self.pop = np.random.rand(self.pop_size, self.indi_len)
        self.skill_factor = np.zeros(self.pop_size, dtype=int)
        self.terminate = False
        self.mfs = None
        self.bfs = None

        #stopping criteria
        self.eval_cnt = 0
        self.max_eval = 0

        for i in range(self.pop_size):
            self.skill_factor[i] = i % self.num_tasks

        self.fitness = np.zeros(self.pop_size)
        self.best_fitness = np.zeros(self.num_tasks)
        self.mean_fitness = np.zeros(self.num_tasks)

        for task_id in range(self.num_tasks):
            task_mask = self.skill_factor == task_id
            self.fitness[task_mask] = self.tasks[task_id].fitness(self.pop[task_mask])
            self.best_fitness[task_id] = np.min(self.fitness[task_mask])
            self.mean_fitness[task_id] = np.mean(self.fitness[task_mask])

        self.armp_matrix = np.full((self.num_tasks, self.num_tasks), 0.3)
        np.fill_diagonal(self.armp_matrix, 1.0)

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

        p1_skill_factor = np.asarray(p1_skill_factor, dtype=int)
        p2_skill_factor = np.asarray(p2_skill_factor, dtype=int)
        
        p1_fitness = self.fitness[p1_indices]
        p2_fitness = self.fitness[p2_indices]
        return self.pop[p1_indices], self.pop[p2_indices], p1_skill_factor, p2_skill_factor, p1_fitness, p2_fitness

    def get_random_individuals(self, num):
        p_indices = np.random.randint(0, self.pop_size, num)
        p = self.pop[p_indices]
        p_skill_factor = self.skill_factor[p_indices]
        p_fitness = self.fitness[p_indices]
        return p, p_skill_factor, p_fitness

    def collect_population_state(self, gen, lookback, success_rate):
        state = {
            "task_count": self.num_tasks,
            "diversity": [],
            "convergence": [],
            "fitness_variance": [],
            "offspring_success_rate": [],
        }   

        state["offspring_success_rate"] = success_rate
        
        for task_id in range(self.num_tasks):
            task_mask = self.skill_factor == task_id
            task_pop = self.pop[task_mask]

            if len(task_pop) > 1:
                distances = [np.linalg.norm(task_pop[i] - task_pop[j]) 
                            for i in range(len(task_pop)) 
                            for j in range(i+1, len(task_pop))]
                diversity = np.mean(distances) if distances else 0.0
            else:
                diversity = 0.0
            state["diversity"].append(diversity)

            task_fitness = self.fitness[task_mask]
            fitness_var = np.var(task_fitness) if len(task_fitness) > 1 else 0.0
            state["fitness_variance"].append(fitness_var)


            if len(self.bfs[task_id]) >= lookback + 1:
                old_fitness = self.bfs[task_id][gen - lookback - 1]
                new_fitness = self.bfs[task_id][gen - 1]
                if old_fitness != 0 and old_fitness != float('inf'):
                    convergence = (old_fitness - new_fitness) / abs(old_fitness)
                else:
                    convergence = 0.0
            else:
                convergence = 0.0
            state["convergence"].append(convergence)

        # print("Diversity:" + str(state["diversity"]))
        # print("Convergence:" + str(state["convergence"]))
        # print("Fitness variance:" + str(state["fitness_variance"]))
        # print("Offspring success rate:" + str(state["offspring_success_rate"]))

        return state

    def get_prob_distrubtion(self):
        #shape = (skill_factor, dimension)
        #shape = (skill_factor, dimension)
        mean = np.zeros(self.num_tasks, self.indi_len)
        var = np.zeros(self.num_tasks, self.indi_len)

        for task_id in range(self.num_tasks):
            task_mask = self.skill_factor == task_id
            g = self.pop[task_mask]
            mean[task_id] = np.mean(g, axis=0)
            var[task_id] = np.var(g, axis=0)

        return mean, var


    def evolve(self, gen, llm_rate, lookback):
        #Crossover
        num_pair = self.pop_size #full
        p1, p2, p1_skill_factor, p2_skill_factor, p1_fitness, p2_fitness = self.get_random_parents(num_pair)

        eval = False
        if gen != 0:
            if gen % llm_rate == 0 or gen == lookback + 1:
                eval = True

        if eval:
            off, off_skill_factor, off_fitness, better_off_cnt = self.crossover(self.armp_matrix, p1, p2, 
                                                                                p1_skill_factor, 
                                                                                p2_skill_factor, 
                                                                                p1_fitness, 
                                                                                p2_fitness,
                                                                                self.tasks,
                                                                                eval)
            total_offspring = 2 * len(p1)
            success_rate = better_off_cnt / total_offspring if total_offspring > 0 else 0.0
        else:
            off, off_skill_factor, off_fitness = self.crossover(self.armp_matrix, p1, p2, 
                                               p1_skill_factor, 
                                               p2_skill_factor, 
                                               p1_fitness, 
                                               p2_fitness,
                                               self.tasks)
        
        #Adaptive RMP
        if gen != 0:
            if gen % llm_rate == 0 or gen == lookback + 1:
                collect_state = self.collect_population_state(gen, lookback, success_rate)
                self.armp_matrix = self.rmp(collect_state, p1, p2, p1_skill_factor, p2_skill_factor, p1_fitness, p2_fitness, gen, lookback, self.tasks)

        ipop = np.concatenate([self.pop, off])
        iskill_factor = np.concatenate([self.skill_factor, off_skill_factor])
        ifitness = np.concatenate([self.fitness, off_fitness])

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
        
    def fit(self, max_eval=1000000, num_gen=5000, monitor=False, monitor_rate=100, llm_rate=100):
        #History Data
        self.max_eval = max_eval
        self.bfs = np.zeros(shape=(self.num_tasks, num_gen + 1))
        self.mfs = np.zeros(shape=(self.num_tasks, num_gen + 1))

        for gen in range(num_gen + 1):
            start_time = time.time()
            self.evolve(gen, llm_rate, lookback=10)
            end_time = time.time()

            if self.terminate:
                print("Out of evaluation!")
                for task_id in range(self.num_tasks):
                        print("Task {0}, Best: {1}, Avg: {2}".format(task_id, self.best_fitness[task_id], self.mean_fitness[task_id]))

                return self.bfs, self.mfs

            for task_id in range(self.num_tasks):
                self.bfs[task_id][gen] = self.best_fitness[task_id]
                self.mfs[task_id][gen] = self.mean_fitness[task_id]

            if gen % monitor_rate == 0:
                print("Gen {0}".format(gen))
                if monitor:
                    print("Evaluation count: {0}".format(self.eval_cnt))
                    for task_id in range(self.num_tasks):
                        print("Task {0}, Best: {1}, Avg: {2}".format(task_id, self.best_fitness[task_id], self.mean_fitness[task_id]))
                print("Time taken each gen: %.4f seconds\n" % (end_time - start_time))

        return self.bfs, self.mfs
