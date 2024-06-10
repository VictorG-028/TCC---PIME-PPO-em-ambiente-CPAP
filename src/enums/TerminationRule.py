from enum import Enum

import numpy as np

def intervals(timestep: int, intervals_sum: int) -> bool:
    return timestep >= intervals_sum

class TerminationRule(Enum):
    INTERVALS = 'difference'                     # TerminationRule.INTERVALS

ALL_TERMINATION_RULE = list( TerminationRule.__members__.values() )

termination_functions = {
    # Name -> Function
    TerminationRule.INTERVALS: intervals,
}
