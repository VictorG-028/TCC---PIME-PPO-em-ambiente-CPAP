from enum import Enum

import numpy as np



class TerminationRule(Enum):
    """Maps Name to lambda (int, int) -> bool"""
    INTERVALS = 1 # lambda timestep, intervals_sum: timestep >= intervals_sum

    # Mapeamento de membros para funções
    __functions = {
        INTERVALS: lambda timestep, intervals_sum: timestep >= intervals_sum
    }

    def __call__(self, timestep: int, intervals_sum: int) -> bool:
        # Chama a função associada ao membro
        return TerminationRule.__functions[self.value](timestep, intervals_sum)




class ErrorFormula(Enum):
    """  """
    DIFFERENCE = 1         # lambda y, y_ref: np.float64(y - y_ref)
    DIFFERENCE_SQUARED = 2 # lambda y, y_ref: np.float64((y - y_ref) * (y - y_ref))

    @staticmethod
    def __difference(y: np.float64, y_ref: np.float64) -> np.float64:
        return (y - y_ref)

    @staticmethod
    def __difference_squared(y: np.float64, y_ref: np.float64) -> np.float64:
        return (y - y_ref) * (y - y_ref)

    # Mapeamento de membros para funções
    __functions = {
        DIFFERENCE: __difference,
        DIFFERENCE_SQUARED: __difference_squared
    }

    def __call__(self, y: np.float64, y_ref: np.float64) -> np.float64:
        return ErrorFormula.__functions[self.value](y, y_ref)




class NamedPoint(Enum):
    X1 = "x1"
    X2 = "x2"
    X3 = "x3"
    X4 = "x4"
