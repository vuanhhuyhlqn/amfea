import numpy as np
from .AbstractRMP import AbstractRMP
from llm import *
from crossover import *
from mutation import *
from dotenv import load_dotenv
import os

load_dotenv()

GPT_API_KEY = os.getenv("GPT_API_KEY")

llm = GPTModel(GPT_API_KEY, "gpt-3.5-turbo-0125", 0.7)
class IndividualRMP:
    def __init__(self, strategy):
        self.strategy = strategy
        self.rmp_matrix = None
        self.performance = None

    def validate_rmp_matrix(self, rmp, task_count):
        if not isinstance(rmp, np.ndarray) or rmp.shape != (task_count, task_count):
            return False
        if not np.all((rmp >= 0) & (rmp <= 1)):
            return False
        if not np.allclose(np.diagonal(rmp), 1.0, atol=1e-6):
            return False
        if not np.allclose(rmp, rmp.T, atol=1e-6):
            return False
        return True

    def fix_rmp_matrix(self, rmp, task_count):
        if not isinstance(rmp, np.ndarray) or rmp.shape != (task_count, task_count):
            rmp = np.full((task_count, task_count), 0.3)
        rmp = np.clip(rmp, 0.0, 1.0)
        rmp = np.maximum(rmp, rmp.T)
        np.fill_diagonal(rmp, 1.0)
        return rmp

    def evaluate(self, collect_state, p1, p2, p1_skill_factor, p2_skill_factor, tasks):
        print("Evaluating strategy")
        print(f"Strategy: {self.strategy}")
        rmp_function = llm.strategy_to_code(self.strategy)
        print(f"RMP function: {rmp_function}")
        try:
            f = {}
            exec(rmp_function, f)
            rmp_matrix = f["get_rmp_matrix"](collect_state["task_count"], 
                                            collect_state["task_performance"], 
                                            collect_state["diversity"],
                                            collect_state["convergence"],
                                            collect_state["task_similarity"])
            rmp_matrix = np.array(rmp_matrix)
            if not self.validate_rmp_matrix(rmp_matrix, len(tasks)):
                print(f"Invalid RMP matrix generated, attempting to fix")
                rmp_matrix = self.fix_rmp_matrix(rmp_matrix, len(tasks))
                if not self.validate_rmp_matrix(rmp_matrix, len(tasks)):
                    print(f"Fixed RMP matrix still invalid, using default")
                    rmp_matrix = np.full((len(tasks), len(tasks)), 0.3)
                    np.fill_diagonal(rmp_matrix, 1.0)
        except Exception as e:
            print(f"Error in creating RMP matrix: {e}")
            rmp_matrix = np.full((len(tasks), len(tasks)), 0.3)
            np.fill_diagonal(rmp_matrix, 1.0)

        self.rmp_matrix = rmp_matrix
        print(self.rmp_matrix)
        mutation = PolynomialMutation(5, 0.02)
        crossover = SBXCrossover(mutation, eta=2)
        _, _, better_off_cnt = crossover(self.rmp_matrix, p1, p2, p1_skill_factor, p2_skill_factor, eval=True, tasks=tasks)
        print(f"Better off count: {better_off_cnt}")
        self.performance = better_off_cnt / len(p1) * 100
        print(f"Performance: {self.performance}")

        return self.performance
class PopulationRMP:
    def __init__(self, pop_size):
        self.pop_size = pop_size
        self.individuals = []

    def gen_pop(self, collect_state, p1, p2, p1_skill_factor, p2_skill_factor, tasks):
        strategies = llm.initial_strategies(self.pop_size)
        for strategy in strategies:
            individual = IndividualRMP(strategy)
            individual.evaluate(collect_state, p1, p2, p1_skill_factor, p2_skill_factor, tasks)
            self.individuals.append(individual)

class AdaptiveRMPMatrix(AbstractRMP):
    def __init__(self, rmp_pop_size, num_gen, pc, pm):
        self.rmp_pop_size = rmp_pop_size
        self.rmp_pop = PopulationRMP(self.rmp_pop_size)
        self.num_gen = num_gen
        self.pc = pc
        self.pm = pm

    def get_rmp(self, collect_state, p1, p2, p1_skill_factor, p2_skill_factor, gen_mfea, lookback, tasks):
        if gen_mfea == lookback + 1:
            self.rmp_pop.gen_pop(collect_state, p1, p2, p1_skill_factor, p2_skill_factor, tasks)
        
        for _ in range(self.num_gen):
            off_list = []
            par1, par2 = np.random.choice(self.rmp_pop.individuals, 2)
            if np.random.rand() < self.pc:
                off_strategy = llm.crossover(par1.strategy, par2.strategy, par1.performance, par2.performance)
                crossover_individual = IndividualRMP(off_strategy)
                individual_performance = crossover_individual.evaluate(collect_state, p1, p2, p1_skill_factor, p2_skill_factor, tasks)
                if crossover_individual.performance >= par1.performance or crossover_individual.performance >= par2.performance:
                    off_list.append(crossover_individual)
                else:
                    off_strategy = llm.reverse(off_strategy)
                    reversed_individual = IndividualRMP(off_strategy)
                    individual_performance = reversed_individual.evaluate(collect_state, p1, p2, p1_skill_factor, p2_skill_factor, tasks)
                    off_list.append(reversed_individual)

                if np.random.rand() < self.pm:
                    off_strategy = llm.mutation(off_strategy, individual_performance)
                    mutation_individual = IndividualRMP(off_strategy)
                    individual_performance = mutation_individual.evaluate(collect_state, p1, p2, p1_skill_factor, p2_skill_factor, tasks)
                off_individual = IndividualRMP(off_strategy)
                off_individual.performance = individual_performance
                off_list.append(off_individual)

            self.rmp_pop.individuals.extend(off_list)
            self.rmp_pop.individuals.sort(key=lambda x: x.performance, reverse=True)
            self.rmp_pop.individuals = self.rmp_pop.individuals[:self.rmp_pop_size]

        best_individual = self.rmp_pop.individuals[0]
        print(best_individual.strategy)
        print(best_individual.rmp_matrix)
        return best_individual.rmp_matrix
            
    def __call__(self, collect_state, p1, p2, p1_skill_factor, p2_skill_factor, gen, llm_rate, tasks):
        return self.get_rmp(collect_state, p1, p2, p1_skill_factor, p2_skill_factor, gen, llm_rate, tasks)