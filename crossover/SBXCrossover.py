import numpy as np
from .AbstractCrossover import AbstractCrossover
from mutation import AbstractMutation

class SBXCrossover(AbstractCrossover):
	def __init__(self, mutation: AbstractMutation, eta=2):
		self.mutation = mutation
		self.eta = eta
		
	def crossover(self, rmp, p1, p2, p1_skill_factor, p2_skill_factor, eval=False, tasks=None):
		assert(len(rmp) == len(p1) and len(rmp) == len(p2))
		rnd = np.random.uniform(size = rmp.shape)
		rnd[p1_skill_factor == p2_skill_factor] = 0.0
		crossover_mask = rnd < rmp
		crossover_indices = np.where(rnd < rmp)[0]

		_p1 = p1[crossover_mask]
		_p2 = p2[crossover_mask]
		
		u = np.random.uniform(size=len(_p1))
		beta = np.zeros(shape=len(_p1))

		mask1 = u < 0.5
		beta[mask1] = (u[mask1] * 2) ** (1 / (self.eta + 1))
		mask2 = u >= 0.5
		beta[mask2] = 1 / (2 * (1 - u[mask2])) ** (1 / (self.eta + 1))

		new_beta_shape = (_p1.shape[0], _p1.shape[1])
		new_beta_strides = (beta.strides[0], 0)

		nbeta = np.lib.stride_tricks.as_strided(beta, new_beta_shape, new_beta_strides)

		off1 = 0.5 * ((1 + nbeta) * _p1 + (1 - nbeta) * _p2)
		off1_skill_factor = p1_skill_factor[crossover_indices]

		off2 = 0.5 * ((1 - nbeta) * _p1 + (1 + nbeta) * _p2)
		off2_skill_factor = p1_skill_factor[crossover_indices]

		off = np.concatenate([off1, off2])
		off = np.clip(off, a_min=0, a_max=1)
		off_skill_factor = np.concatenate([off1_skill_factor, off2_skill_factor])

		mutation_mask = np.invert(crossover_mask)
		_p1_mutation = p1[mutation_mask]
		_p2_mutation = p2[mutation_mask]
		off_mut_1, off_mut_skill_factor_1 = self.mutation(_p1_mutation, p1_skill_factor[mutation_mask])
		off_mut_2, off_mut_skill_factor_2 = self.mutation(_p2_mutation, p2_skill_factor[mutation_mask])

		off = np.concatenate([off, off_mut_1, off_mut_2])
		off_skill_factor = np.concatenate([off_skill_factor, off_mut_skill_factor_1, off_mut_skill_factor_2])

		if eval:
			off_fitness = np.zeros(shape=len(off))
			num_tasks = len(tasks)
			better_off_cnt = 0
	
			for task_id in range(num_tasks):
				task_mask = off_skill_factor == task_id
				off_fitness[task_mask] = tasks[task_id].fitness(off[task_mask]) 
				p_fitness = tasks[task_id].fitness(_p1[task_mask])
				# print(off_fitness[task_mask])
				# print(p_fitness)
				better_off_cnt += np.sum(off_fitness[task_mask] > p_fitness)
			return off, off_skill_factor, better_off_cnt

		assert(len(off) == len(off_skill_factor))
		return off, off_skill_factor

	def __call__(self, rmp, p1, p2, p1_skill_factor, p2_skill_factor, eval=False, tasks=None):
		return self.crossover(rmp, p1, p2, p1_skill_factor, p2_skill_factor, eval, tasks)
