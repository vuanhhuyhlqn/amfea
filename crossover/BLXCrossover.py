import numpy as np
from .AbstractCrossover import AbstractCrossover

class BLXCrossover(AbstractCrossover):
	def __init__(self, alpha=0.5):
		self.alpha = alpha

	def crossover(self, rmp, p1, p2, p1_skill_factor, p2_skill_factor):
		assert(len(rmp) == len(p1) and len(rmp) == len(p2))
		rnd = np.random.uniform(size = rmp.shape)
		crossover_indices = np.where(rnd < rmp)[0]

		_p1 = p1[crossover_indices]
		_p2 = p2[crossover_indices]
		
		d = np.abs(_p1 - _p2)
		low_bounds = np.minimum(_p1, _p2) - self.alpha * d
		high_bounds = np.maximum(_p1, _p2) + self.alpha * d

		off = np.random.uniform(low_bounds, high_bounds, size=_p1.shape)
		off_skill_factor = p1_skill_factor[crossover_indices]
		assert(len(off) == len(off_skill_factor))
		return off, off_skill_factor

	def __call__(self, rmp, p1, p2, p1_skill_factor, p2_skill_factor):
		return self.crossover(rmp, p1, p2, p1_skill_factor, p2_skill_factor)
