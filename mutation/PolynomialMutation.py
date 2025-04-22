import numpy as np
from .AbstractMutation import AbstractMutation

class PolynomialMutation(AbstractMutation):
	def __init__(self, eta_m, bound):
		self.eta_m = eta_m
		self.bound = bound
	def polynomial_mutation(self, p, p_skill_factor):
		n_pop, n_var = p.shape
		lower_bound = -self.bound
		upper_bound = self.bound
		mutated_p = p.copy()
		u = np.random.random((n_pop, n_var))
		delta = np.where(
			u < 0.5,
			(2 * u) ** (1 / (self.eta_m + 1)) - 1,
			1 - (2 * (1 - u)) ** (1 / (self.eta_m + 1))
		)
		delta_max = np.minimum(
			mutated_p - lower_bound,
			upper_bound - mutated_p
		)
		mutated_p += delta * delta_max
		mutated_p = np.clip(mutated_p, lower_bound, upper_bound)
		return mutated_p, p_skill_factor
	def __call__(self, p, p_skill_factor):
		return self.polynomial_mutation(p, p_skill_factor)
