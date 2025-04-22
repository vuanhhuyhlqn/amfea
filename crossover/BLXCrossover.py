import numpy as np
from .AbstractCrossover import AbstractCrossover

class BLXCrossover(AbstractCrossover):
	def __init__(self, alpha=0.5):
		self.alpha = alpha
		
	def crossover(self, rmp, p1, p2, p1_skill_factor, p2_skill_factor, eval=False, tasks=None):
		assert(len(rmp) == len(p1) and len(rmp) == len(p2))
		assert(rmp.shape == p1_skill_factor.shape)
		assert(rmp.shape == p2_skill_factor.shape)

		rnd = np.random.uniform(size = rmp.shape)
		rnd[p1_skill_factor == p2_skill_factor] = 0.0
		crossover_indices = np.where(rnd < rmp)[0]

		_p1 = p1[crossover_indices]
		_p2 = p2[crossover_indices]
		
		d = np.abs(_p1 - _p2)
		low_bounds = np.minimum(_p1, _p2) - self.alpha * d
		high_bounds = np.maximum(_p1, _p2) + self.alpha * d

		off = np.random.uniform(low_bounds, high_bounds, size=_p1.shape)
		off_skill_factor = p1_skill_factor[crossover_indices]
    
		if eval:
			off_fitness = np.zeros(shape=len(off))
			num_tasks = len(tasks)
			better_off_cnt = 0
	
			for task_id in range(num_tasks):
				task_mask = off_skill_factor == task_id
				off_fitness[task_mask] = tasks[task_id].fitness(off[task_mask]) 
				p_fitness = tasks[task_id].fitness(_p1[task_mask])
				print(off_fitness[task_mask])
				print(p_fitness)
				better_off_cnt += np.sum(off_fitness[task_mask] > p_fitness)
			return off, off_skill_factor, better_off_cnt

		assert(len(off) == len(off_skill_factor))
		return off, off_skill_factor

	def __call__(self, rmp, p1, p2, p1_skill_factor, p2_skill_factor, eval=False, tasks=None):
		return self.crossover(rmp, p1, p2, p1_skill_factor, p2_skill_factor, eval, tasks)
