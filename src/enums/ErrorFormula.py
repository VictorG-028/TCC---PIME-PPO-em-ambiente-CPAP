from enum import Enum

import numpy as np


def difference(y: np.float64, y_ref: np.float64) -> np.float64:
    return (y_ref - y)

def difference_squared(y: np.float64, y_ref: np.float64) -> np.float64:
    return (y_ref - y) * (y_ref - y)


class ErrorFormula(Enum):
    DIFFERENCE = 'difference'                     # ErrorFormula.DIFFERENCE
    DIFFERENCE_SQUARED = 'difference squared'     # ErrorFormula.DIFFERENCE_SQUARED

ALL_ERROR_FORMULAS = list( ErrorFormula.__members__.values() )

error_functions = {
    # Name -> Function
    ErrorFormula.DIFFERENCE: difference,
    ErrorFormula.DIFFERENCE_SQUARED: difference_squared,
}

