from .AbstractTask import AbstractTask
import numpy as np

class CEC17Task(AbstractTask):
    def __init__(self, id, bound):
        self.id = id
        self.bound = bound
    
    def __call__(self, p):
        pass    

def get_10_tasks():
    pass