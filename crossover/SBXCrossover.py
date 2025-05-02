import numpy as np
from .AbstractCrossover import AbstractCrossover
from task import AbstractTask
from mutation import AbstractMutation

class SBXCrossover(AbstractCrossover):
	def __init__(self, mutation: AbstractMutation, eta=2):
		self.mutation = mutation
		self.eta = eta

	def evaluate(self, p, p_skill_factor, tasks):
		p_fitness = np.zeros(len(p))
		for task_id in range(len(tasks)):
			task_mask = p_skill_factor == task_id
			p_fitness[task_mask] = tasks[task_id].fitness(p[task_mask])
		return p_fitness

	def rmp_matrix_to_array(self, rmp_matrix, p1_skill_factor, p2_skill_factor):
		return rmp_matrix[p1_skill_factor, p2_skill_factor]
		
	def crossover(self, 
			   rmp_matrix, 
			   p1, 
			   p2, 
			   p1_skill_factor, 
			   p2_skill_factor, 
			   p1_fitness, 
			   p2_fitness, 
			   tasks, 
			   eval=False):
		rmp = self.rmp_matrix_to_array(rmp_matrix, p1_skill_factor, p2_skill_factor)
		assert(len(rmp) == len(p1) and len(rmp) == len(p2))
		rnd = np.random.rand(len(rmp))
		rnd[p1_skill_factor == p2_skill_factor] = 0.0
		crossover_mask = rnd < rmp

		total_performance_diff = 0
		avg_performance_diff = 0

		_p1 = p1[crossover_mask]
		_p2 = p2[crossover_mask]
		
		u = np.random.rand(len(_p1), len(_p1[0]))
		beta = np.zeros(_p1.shape)

		mask1 = u < 0.5
		beta[mask1] = (u[mask1] * 2) ** (1 / (self.eta + 1))
		mask2 = np.invert(mask1)
		beta[mask2] = (2 * (1 - u[mask2])) ** ((-1) / (self.eta + 1))

		off1 = 0.5 * ((1 + beta) * _p1 + (1 - beta) * _p2)
		off2 = 0.5 * ((1 - beta) * _p1 + (1 + beta) * _p2)

		off1 = np.clip(off1, a_min=0, a_max=1)
		off2 = np.clip(off2, a_min=0, a_max=1)
		
		assert(np.max(off1) <= 1.0)
		assert(np.max(off2) <= 1.0)
		
		off1_skill_factor = p1_skill_factor[crossover_mask]
		off2_skill_factor = p2_skill_factor[crossover_mask]
		off1_fitness = self.evaluate(off1, off1_skill_factor, tasks)
		off2_fitness = self.evaluate(off2, off2_skill_factor, tasks)

		if eval:
			p_fitness = p1_fitness[crossover_mask]
			assert(len(p_fitness) == len(off1_fitness))
			assert(len(p_fitness) == len(off2_fitness))

			assert(np.min(p_fitness) > 0)
			diff1 = p_fitness - off1_fitness
			diff_percentage1 = (diff1 / p_fitness) * 100

			diff2 = p_fitness - off2_fitness
			diff_percentage2 = (diff2 / p_fitness) * 100

			total_performance_diff += np.sum(diff_percentage1) + np.sum(diff_percentage2)

		off = np.concatenate([off1, off2])
		off_skill_factor = np.concatenate([off1_skill_factor, off2_skill_factor])
		off_fitness = np.concatenate([off1_fitness, off2_fitness])

		mutation_mask = np.invert(crossover_mask)
		# print(mutation_mask)
		_p1_mutation = p1[mutation_mask]
		_p2_mutation = p2[mutation_mask]
		off_mut_1, off_mut_skill_factor_1 = self.mutation(_p1_mutation, p1_skill_factor[mutation_mask])
		off_mut_2, off_mut_skill_factor_2 = self.mutation(_p2_mutation, p2_skill_factor[mutation_mask])
		
		off_mut_fitness_1 = self.evaluate(off_mut_1, off_mut_skill_factor_1, tasks)
		off_mut_fitness_2 = self.evaluate(off_mut_1, off_mut_skill_factor_2, tasks)

		if eval:
			p1_fitness = p1_fitness[mutation_mask]
			p2_fitness = p2_fitness[mutation_mask]

			diff1 = p1_fitness - off_mut_fitness_1
			diff2 = p2_fitness - off_mut_fitness_2

			if p1_fitness.size > 0:
				assert(np.min(p1_fitness) > 0)
				diff_percentage1 = (diff1 / p1_fitness) * 100
				total_performance_diff += np.sum(diff_percentage1) 

			if p2_fitness.size > 0:
				assert(np.min(p2_fitness) > 0)
				diff_percentage2 = (diff2 / p2_fitness) * 100
				total_performance_diff += np.sum(diff_percentage2) 

		off = np.concatenate([off, off_mut_1, off_mut_2])
		off_skill_factor = np.concatenate([off_skill_factor, off_mut_skill_factor_1, off_mut_skill_factor_2])
		off_fitness = np.concatenate([off_fitness, off_mut_fitness_1, off_mut_fitness_2])
		assert(len(off) == len(off_skill_factor))
		assert(len(off) == len(off_fitness))

		assert(len(off) > 0)
		avg_performance_diff = total_performance_diff / (len(off))
		if eval:
			return off, off_skill_factor, off_fitness, avg_performance_diff
		return off, off_skill_factor, off_fitness

	def __call__(self, rmp, 
			  p1, 
			  p2, 
			  p1_skill_factor, 
			  p2_skill_factor, 
			  p1_fitness, 
			  p2_fitness, 
			  tasks, 
			  eval=False):
		return self.crossover(rmp, 
						p1, 
						p2, 
						p1_skill_factor, 
						p2_skill_factor, 
						p1_fitness, 
						p2_fitness, 
						tasks, 
						eval)
