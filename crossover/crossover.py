import numpy as np

class AbstractCrossover:
	def __init__(self):
		pass
	def __call__(self, rmp, p1, p2, p1_skill_factor, p2_skill_factor):
		pass

class ArithmeticCrossover(AbstractCrossover):
	def __init__(self):
		pass

	def crossover(self, rmp, p1, p2, p1_skill_factor, p2_skill_factor):
		assert(len(rmp) == len(p1) and len(rmp) == len(p2))
		rnd = np.random.uniform(size = rmp.shape)
		crossover_indices = np.where(rnd < rmp)[0]
		alpha = np.random.uniform(size=(len(crossover_indices), p1.shape[1]))
		off = p1[crossover_indices] * alpha + p2[crossover_indices] * (1 - alpha)
		off_skill_factor = p1_skill_factor[crossover_indices]
		assert(len(off) == len(off_skill_factor))
		return off, off_skill_factor

	def __call__(self, rmp, p1, p2, p1_skill_factor, p2_skill_factor):
		return self.crossover(rmp, p1, p2)
