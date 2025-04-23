import numpy as np
from .AbstractRMP import AbstractRMP
from llm import *
from crossover import *
from dotenv import load_dotenv
import os

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

deepseek = DeepseekModel(DEEPSEEK_API_KEY, "deepseek-chat", 1.0)
class IndividualRMP:
    def __init__(self, idea):
        self.idea = idea
        self.code = None
        self.performance = None

    def evaluate(self, p1, p2, p1_skill_factor, p2_skill_factor, p1_fitness, p2_fitness, tasks):
        rmp_function = deepseek.idea_to_code_function(self.idea)
        rmp_function = "import numpy as np\n" + rmp_function
        try:
            f = {}
            exec(rmp_function, f)
            rmp = f["get_rmp"](p1, p2, p1_skill_factor, p2_skill_factor, p1_fitness, p2_fitness)
            rmp = np.array(rmp)
            self.code = rmp_function
        except Exception as e:
            print(f"Error in create rmp array: {e}")
            rmp = np.full(len(p1), 0.3)

        crossover = BLXCrossover()
        _, _, better_off_cnt = crossover(rmp, p1, p2, p1_skill_factor, p2_skill_factor, eval=True, tasks=tasks)
        print(f"Better off count: {better_off_cnt}")
        self.performance = better_off_cnt / len(p1)
        print(f"Performance: {self.performance}")

        return self.performance
class PopulationRMP:
    def __init__(self, pop_size):
        self.pop_size = pop_size
        self.individuals = []

    def gen_pop(self, p1, p2, p1_skill_factor, p2_skill_factor, p1_fitness, p2_fitness, tasks):
        ideas = deepseek.initial_ideas(self.pop_size)
        for idea in ideas:
            individual = IndividualRMP(idea)
            individual.evaluate(p1, p2, p1_skill_factor, p2_skill_factor, p1_fitness, p2_fitness, tasks)
            self.individuals.append(individual)

class AdaptiveRMP(AbstractRMP):
    def __init__(self, rmp_pop_size, num_gen, pc, pm):
        self.rmp_pop_size = rmp_pop_size
        self.rmp_pop = PopulationRMP(self.rmp_pop_size)
        self.num_gen = num_gen
        self.pc = pc
        self.pm = pm
        self.function = None

    def get_rmp(self, p1, p2, p1_skill_factor, p2_skill_factor, p1_fitness, p2_fitness, gen_mfea, llm_rate, tasks):
        if gen_mfea % llm_rate == 0:
            if gen_mfea == 0:
                self.rmp_pop.gen_pop(p1, p2, p1_skill_factor, p2_skill_factor, p1_fitness, p2_fitness, tasks)
            off_list = []
            par1, par2 = np.random.choice(self.rmp_pop.individuals, 2)
            if np.random.rand() < self.pc:
                off_idea = deepseek.crossover(par1.idea, par2.idea, par1.performance, par2.performance)
                crossover_individual = IndividualRMP(off_idea)
                individual_performance = crossover_individual.evaluate(p1, p2, p1_skill_factor, p2_skill_factor, p1_fitness, p2_fitness, tasks)
                if crossover_individual.performance >= par1.performance or crossover_individual.performance >= par2.performance:
                    off_list.append(crossover_individual)
                else:
                    off_idea = deepseek.reverse(off_idea)
                    reversed_individual = IndividualRMP(off_idea)
                    individual_performance = reversed_individual.evaluate(p1, p2, p1_skill_factor, p2_skill_factor, p1_fitness, p2_fitness, tasks)
                    off_list.append(reversed_individual)

                if np.random.rand() < self.pm:
                    off_idea = deepseek.mutation(off_idea, off_individual.performance)
                    mutation_individual = IndividualRMP(off_idea)
                    individual_performance = mutation_individual.evaluate(p1, p2, p1_skill_factor, p2_skill_factor, p1_fitness, p2_fitness, tasks)

                off_individual = IndividualRMP(off_idea)
                off_individual.performance = individual_performance
                off_list.append(off_individual)

            self.rmp_pop.individuals.extend(off_list)
            self.rmp_pop.individuals.sort(key=lambda x: x.performance, reverse=True)
            self.rmp_pop.individuals = self.rmp_pop.individuals[:self.rmp_pop_size]

            best_individual = self.rmp_pop.individuals[0]
            self.function = best_individual.code
        
        f = {}
        exec(self.function, f)
        rmp = f["get_rmp"](p1, p2, p1_skill_factor, p2_skill_factor, p1_fitness, p2_fitness)
        return rmp
    
    def __call__(self, p1, p2, p1_skill_factor, p2_skill_factor, p1_fitness, p2_fitness, gen, llm_rate, tasks):
        return self.get_rmp(p1, p2, p1_skill_factor, p2_skill_factor, p1_fitness, p2_fitness, gen, llm_rate, tasks)