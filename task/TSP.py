from .AbstractTask import AbstractTask
from .tsplib import *
import numpy as np

class TSPTask(AbstractTask):
    def __init__(self, problem_name):
        prob = Problem('tsp', problem_name)
        self.num_cities = prob.problem_size
        self.dist_mat = prob.distance_matrix

    def decode(self, p):
        decoded_p = np.argsort(p)
        # print(decoded_p)
        return decoded_p

    def fitness(self, p):
        dp = self.decode(p[:, :self.num_cities])
        distances = np.sum(self.dist_mat[dp[:, 1:], dp[:, :-1]], axis=1)
        np.add(distances, self.dist_mat[0, dp[:, 0]], out=distances)
        np.add(distances, self.dist_mat[dp[:, -1], 0], out=distances)
        return distances