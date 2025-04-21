import numpy as np

class AbstractCrossover:
	def __init__(self):
		pass
	def __call__(self, rmp, p1, p2):
		pass

class ArithmeticCrossover(AbstractCrossover):
	def __init__(self):
		pass

	def crossover(self, rmp, p1, p2):
		assert(len(rmp) == len(p1) and len(rmp) == len(p2))
		rnd = np.random.uniform(size = rmp.shape)
		crossover_indices = np.where(rnd < rmp)[0]
		# print(rnd)
		# print(rmp)
		alpha = np.random.uniform(size=(len(crossover_indices), p1.shape[1]))
		# print(alpha)
		off = p1[crossover_indices] * alpha + p2[crossover_indices] * (1 - alpha)
		return off


	def __call__(self, rmp, p1, p2):
		return self.crossover(rmp, p1, p2)
