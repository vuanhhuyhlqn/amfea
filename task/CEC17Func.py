import numpy as np
from .AbstractFunc import AbstractFunc

class Sphere(AbstractFunc):
    def __init__(self, dim, shift, rotation_matrix):
        super().__init__(dim, shift, rotation_matrix)

    def __call__(self, x):
        _x = self.shift_rotation_decode(x)
        return np.sum(_x**2, axis=0)

class Weierstrass(AbstractFunc):
    def __init__(self, dim, shift, rotation_matrix):
        super().__init__(dim, shift, rotation_matrix)
        self.params = {}
        self.params['a'] = 0.5
        self.params['b'] = 3
        self.params['k_max'] = 21
    
    def __call__(self, x):
        _x = self.shift_rotation_decode(x)
        left = 0
        for i in range(self.dim):
            left += np.sum(self.params['a'] ** np.arange(self.params['k_max']) * np.cos(2*np.pi * self.params['b'] ** np.arange(self.params['k_max']) * (_x[i] + 0.5)))
        right = self.dim * np.sum(self.params['a'] ** np.arange(self.params['k_max']) * np.cos(2 * np.pi * self.params['b'] ** np.arange(self.params['k_max']) * 0.5))
        return left - right
    
class Ackley(AbstractFunc):
    def __init__(self, dim, shift, rotation_matrix):
        super().__init__(dim, shift, rotation_matrix)
        self.params = {}
        self.params['a'] = 20
        self.params['b'] = 0.2
        self.params['c'] = 2 * np.pi
    def __call__(self, x):
        _x = self.shift_rotation_decode(x)
        return -self.params['a'] * np.exp(-self.params['b']*np.sqrt(np.mean(_x**2)))\
                - np.exp(np.mean(np.cos(self.params['c'] * _x)))\
                + self.params['a']\
                + np.exp(1)
    
class Rosenbrock(AbstractFunc):
    def __init__(self, dim, shift, rotation_matrix):
        super().__init__(dim, shift, rotation_matrix)
    def __call__(self, x):
        _x = self.shift_rotation_decode(x)
        l = 100*np.sum((_x[1:] - _x[:-1]**2) ** 2)
        r = np.sum((_x[:-1] - 1) ** 2)
        return l + r

class Schwefel(AbstractFunc):
    def __init__(self, dim, shift, rotation_matrix):
        super().__init__(dim, shift, rotation_matrix)
    def __call__(self, x):
        _x = self.shift_rotation_decode(x)
        return 418.9829 * self.dim - np.sum(_x * np.sin(np.sqrt(np.abs(_x))))

class Griewank(AbstractFunc):
    def __init__(self, dim, shift, rotation_matrix):
        super().__init__(dim, shift, rotation_matrix)
    def __call__(self, x):
        _x = self.shift_rotation_decode(x)
        return np.sum(_x**2) / 4000 - np.prod(np.cos(_x / np.sqrt((np.arange(self.dim) + 1)))) + 1

class Rastrigin(AbstractFunc):
    def __init__(self, dim, shift, rotation_matrix):
        super().__init__(dim, shift, rotation_matrix)
    def __call__(self, x):
        _x = self.shift_rotation_decode(x)
        return 10 * self.dim + np.sum(_x ** 2 - 10 * np.cos(2 * np.pi * _x))