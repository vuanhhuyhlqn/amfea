import numpy as np
from .AbstractRMP import AbstractRMP
from llm import *
# from dotenv import load_dotenv
import os

# load_dotenv()

DEEPSEEK_API_KEY = "sk-505a4ff57cfb432d8888a3d8d66a3133"

deepseek = DeepseekModel(DEEPSEEK_API_KEY, "deepseek-chat", 1.0)
class IndividualRMP:
    def __init__(self, idea):
        self.idea = idea
        self.performance = None

    def evaluate(self, p1, p2, p1_skill_factor, p2_skill_factor, p1_fitness, p2_fitness):
        while True:
            rmp_function = deepseek.idea_to_code_function(self.idea)
            try:
                f = {}
                exec(rmp_function, f)
                rmp = f["rmp"](p1, p2, p1_skill_factor, p2_skill_factor, p1_fitness, p2_fitness)
            except Exception as e:
                print(f"Error in create rmp array: {e}")

    
class PopulationRMP:
    def __init__(self, pop_size):
        self.pop_size = pop_size
        self.individuals = []

    def gen_pop(self, p1, p2, p1_skill_factor, p2_skill_factor, p1_fitness, p2_fitness):
        ideas = deepseek.initial_ideas(self.pop_size)
        for idea in ideas:
            individual = IndividualRMP(idea)
            self.individuals.append(individual)

class AdaptiveRMP(AbstractRMP):
    def __init__(self):
        pass

    def get_rmp(self, size):
        pass
    
    def __call__(self, p1, p2, p1_skill_factor, p2_skill_factor, p1_fitness, p2_fitness):
        return self.get_rmp(p1, p2, p1_skill_factor, p2_skill_factor, p1_fitness, p2_fitness)