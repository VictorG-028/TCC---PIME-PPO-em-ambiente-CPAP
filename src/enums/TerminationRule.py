from enum import Enum

import numpy as np

def intervals(timestep: int, intervals_sum: int, max_step: int) -> bool:
    return timestep >= intervals_sum

def max_steps(timestep: int, intervals_sum: int, max_step: int) -> bool:
    return timestep >= max_step

class TerminationRule(Enum):
    INTERVALS = 'intervals'                     # TerminationRule.INTERVALS
    MAX_STEPS = 'max_steps'                     # TerminationRule.MAX_STEPS

ALL_TERMINATION_RULE = list( TerminationRule.__members__.values() )

termination_functions = {
    # Name -> Function
    TerminationRule.INTERVALS: intervals,
    TerminationRule.MAX_STEPS: max_steps,
}
