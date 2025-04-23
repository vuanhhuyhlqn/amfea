import numpy as np
from .AbstractRMP import AbstractRMP
from llm import *
from crossover import *
import os

class Population:
    def __init__(self, pop_size, num_gen):
        self.pop_size = pop_size
        self.num_gen = num_gen
        self.pop = []
        
        DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
        self.model = DeepsekModel2(DEEPSEEK_API_KEY)

        for _ in range(self.pop_size):
            self.pop.append(self.model.init_idea_from_llm())
        
        for p in self.pop:
            print(p)
    
    def evaluate(self, p, p1, p2, p1_skill_factor, p2_skill_factor, p1_fitness, p2_fitness):
        rmp_function = self.model.gen_code_from_idea(p)
        f = {}
        exec(rmp_function, f)
        rmp = f["get_rmp"](p1, p2, p1_skill_factor, p2_skill_factor)
        print(rmp)

class ApdaptiveRMP2(AbstractRMP):
    def __init__(self):
        pass
    def __call__(self, p1, p2, p1_skill_factor, p2_skill_factor):
        pass