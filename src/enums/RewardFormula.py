from enum import Enum

import numpy as np


def difference(y: np.float64, y_ref: np.float64) -> np.float64:
    return (y - y_ref) # Same as -(y_ref - y)

def difference_squared(y: np.float64, y_ref: np.float64) -> np.float64:
    return - ((y_ref - y) * (y_ref - y))


class RewardFormula(Enum):
    DIFFERENCE = 'difference'                     # RewardFormula.DIFFERENCE
    DIFFERENCE_SQUARED = 'difference squared'     # RewardFormula.DIFFERENCE_SQUARED

ALL_Reward_FORMULAS = list( RewardFormula.__members__.values() )

reward_functions = {
    # Name -> Function
    RewardFormula.DIFFERENCE: difference,
    RewardFormula.DIFFERENCE_SQUARED: difference_squared,
}

