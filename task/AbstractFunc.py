import numpy as np

class AbstractFunc:
    def __init__(self, dim, shift = None, rotation_matrix = None):
        self.dim = dim
        self.shift = shift
        if rotation_matrix is not None:
            self.rotation_matrix = rotation_matrix
        else:
            self.rotation_matrix = np.identity(dim)
        
        if shift is not None:
            self.shift = shift
        else:
            self.shift = np.zeros(dim)
    def shift_rotation_decode(self, x):
        print(x.shape)
        print(self.shift.shape)
        print(self.rotation_matrix.shape)
        return self.rotation_matrix @ (x - self.shift)
    def __call__(self, x):
        pass