import numpy as np
from .AbstractMutation import AbstractMutation

class PolynomialMutation(AbstractMutation):
	def __init__(self, eta_m, p_m):
		self.eta_m = eta_m
		self.p_m = p_m

	def polynomial_mutation(self, p, p_skill_factor):
		mutated_p = p.copy()
		
		rnd = np.random.rand(p.shape[0], p.shape[1])
		mask = rnd < self.p_m

		u = np.random.rand(p.shape[0], p.shape[1])
		delta = np.where(
			u < 0.5,
			(2 * u) ** (1 / (self.eta_m + 1)) - 1,
			1 - (2 * (1 - u)) ** (1 / (self.eta_m + 1))
		)

		mask1 = mask & (delta < 0)
		mask2 = mask & (delta >= 0)
		mutated_p[mask1] += delta[mask1] * mutated_p[mask1]
		mutated_p[mask2] += delta[mask2] * (1 - mutated_p[mask2]) 

		mutated_p = np.clip(mutated_p, 0, 1)
		assert(np.max(mutated_p) <= 1.0)
		return mutated_p, p_skill_factor
	def __call__(self, p, p_skill_factor):
		return self.polynomial_mutation(p, p_skill_factor)
