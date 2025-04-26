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
                 optimums: List[float] = None
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
        self.optimums = optimums
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

        self.max_fitness_distances = self.mean_fitness - self.optimums
        self.max_fitness_distances = np.where(self.max_fitness_distances == 0, 1e-10, self.max_fitness_distances) 

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

    def collect_population_state(self, gen, lookback):
        state = {
            "task_count": self.num_tasks,
            "task_performance": [],
            "diversity": [],
            "convergence": [],
            "task_similarity": []
        }

        task_performance = (self.max_fitness_distances - (self.mean_fitness - self.optimums)) / self.max_fitness_distances * 100
        task_performance = np.clip(task_performance, 0, 100)
        state["task_performance"] = task_performance.tolist()        
        print("Task Performance:" + str(state["task_performance"])) 
        
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

        print("Diversity:" + str(state["diversity"]))
        print("Convergence:" + str(state["convergence"]))

        for i in range(self.num_tasks):
            for j in range(i+1, self.num_tasks):
                mask_i = self.skill_factor == i
                mask_j = self.skill_factor == j
                fitness_i = self.fitness[mask_i]
                pop_i = self.pop[mask_i]
                pop_j = self.pop[mask_j]
                fitness_j = self.fitness[mask_j]

                fitness_i_on_j = self.tasks[j].fitness(pop_i)
                fitness_j_on_i = self.tasks[i].fitness(pop_j)

                if len(fitness_i) == len(fitness_i_on_j) and len(fitness_j) == len(fitness_j_on_i):
                    corr_i, _ = pearsonr(fitness_i, fitness_i_on_j) if len(fitness_i) > 1 else (0.0, 0.0)
                    corr_j, _ = pearsonr(fitness_j, fitness_j_on_i) if len(fitness_j) > 1 else (0.0, 0.0)
                    fitness_corr = max(corr_i, corr_j, 0.0)
                else:
                    fitness_corr = 0.0

                best_idx_i = np.argmin(fitness_i) if len(fitness_i) > 0 else 0
                best_idx_j = np.argmin(fitness_j) if len(fitness_j) > 0 else 0
                best_ind_i = pop_i[best_idx_i] if len(pop_i) > 0 else np.zeros(self.indi_len)
                best_ind_j = pop_j[best_idx_j] if len(pop_j) > 0 else np.zeros(self.indi_len)
                solution_distance = np.linalg.norm(best_ind_i - best_ind_j)
                max_distance = np.sqrt(self.indi_len)
                solution_similarity = np.clip(1.0 - (solution_distance / max_distance), 0.0, 1.0)

                similarity = 0.5 * fitness_corr + 0.5 * solution_similarity
                state["task_similarity"].append((i, j, similarity))

        print("Task Similarity:" + str(state["task_similarity"]))

        return state

    def evolve(self, gen, llm_rate):
        # num_pair = np.random.randint(int(self.pop_size * 9 / 10), int(self.pop_size))
        num_pair = self.pop_size #full
        p1, p2, p1_skill_factor, p2_skill_factor, p1_fitness, p2_fitness = self.get_random_parents(num_pair)
        #Adaptive RMP
        
        if gen % llm_rate == 0 and gen != 0:
            collect_state = self.collect_population_state(gen, lookback=10)
            armp = self.rmp(p1, p2, p1_skill_factor, p2_skill_factor, p1_fitness, p2_fitness, gen, llm_rate, self.tasks)
        else:
            normalRMP = NormalRMP()
            armp = normalRMP(p1, p2, p1_skill_factor, p2_skill_factor, p1_fitness, p2_fitness, gen, llm_rate, self.tasks)
        
        #Crossover 
        off, off_skill_factor = self.crossover(armp, p1, p2, p1_skill_factor, p2_skill_factor)
        off_fitness = np.zeros(len(off), dtype=np.float32)

        #Calculate crossover offsprings fitness
        for task_id in range(self.num_tasks):
            task_mask = off_skill_factor == task_id 
            off_fitness[task_mask] = self.tasks[task_id].fitness(off[task_mask])

            if self.eval_cnt + len(off[task_mask]) > self.max_eval:
                self.terminate = True
                return

            self.eval_cnt += len(off[task_mask])

        #Mutation
        # num_mutation = np.random.randint(0, int(self.pop_size) * 15 / 100)
        # off_mut, off_mut_skill_factor, off_mut_fitness = self.get_random_individuals(num_mutation)
        # off_mut, off_mut_skill_factor = self.mutation(off_mut, off_mut_skill_factor)
        
        #Calculate mutation offsprings fitness
        # for task_id in range(self.num_tasks):
        #     task_mask = off_mut_skill_factor == task_id 
        #     off_mut_fitness[task_mask] = self.tasks[task_id].fitness(off_mut[task_mask])

        #     if self.eval_cnt + len(off_mut[task_mask]) > self.max_eval:
        #         self.terminate = True
        #         return

        #     self.eval_cnt += len(off_mut[task_mask])
        
        # ipop = np.concatenate([self.pop, off, off_mut])
        # iskill_factor = np.concatenate([self.skill_factor, off_skill_factor, off_mut_skill_factor])
        # ifitness = np.concatenate([self.fitness, off_fitness, off_mut_fitness])

        ipop = np.concatenate([self.pop, off])
        iskill_factor = np.concatenate([self.skill_factor, off_skill_factor])
        ifitness = np.concatenate([self.fitness, off_fitness])

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
        
    def fit(self, max_eval=1000000, num_gen=5000, monitor=False, monitor_rate=100, llm_rate=100):
        #History Data
        self.max_eval = max_eval
        self.bfs = np.zeros(shape=(self.num_tasks, num_gen + 1))
        self.mfs = np.zeros(shape=(self.num_tasks, num_gen + 1))

        for gen in range(num_gen + 1):
            start_time = time.time()
            self.evolve(gen, llm_rate)
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
