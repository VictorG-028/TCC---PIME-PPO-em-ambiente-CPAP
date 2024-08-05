from environments.base_set_point_env import BaseSetPointEnv
from typing import Callable, Literal, Optional
from enums.ErrorFormula import ErrorFormula, error_functions
from enums.TerminationRule import TerminationRule, termination_functions
from modules.Scheduller import Scheduller
from modules.Ensemble import Ensemble

import numpy as np
import gymnasium
from gymnasium import spaces



class PhControl(BaseSetPointEnv):
    """
    Environment specific for residual water treatment by controling ph.
    This class defines state and action spaces.
    """

    def __init__(
            self,

            # Base class parameters
            scheduller: Scheduller,
            simulation_model: Callable,
            ensemble_params: dict[str, np.float64],
            x_start_points: Optional[list[np.float64]] = None,
            termination_rule: TerminationRule | Callable = TerminationRule.INTERVALS,
            error_formula: ErrorFormula | Callable = ErrorFormula.DIFFERENCE_SQUARED,
            render_mode: Literal["terminal"] = "terminal",
            ):

        assert x_start_points is None or 2 == len(x_start_points), \
            "Lenght of start_points must be equal to 2."
        # assert action_size == 1, \
        #     "This env have only 1 output action. action_size must be 1."
        # assert tracked_point == 'x1', \
        #     "This env tracks the ph (x1). tracked_point must be 'x1'."

        super().__init__(
            scheduller=scheduller,
            simulation_model=simulation_model,
            ensemble_params=ensemble_params,
            termination_rule=termination_rule,
            error_formula=error_formula,
            action_size=1,
            x_size=2,
            x_start_points=x_start_points,
            tracked_point="x1",
            extra_inputs={'dt': 20},
            render_mode=render_mode,
        )

        # Definindo o espaço de ações (u_t)
        self.action_space = spaces.Box(
            low=-1_000, # TODO/Perguntar: validar low e high
            high=1_000, 
            shape=(1,),
            dtype=np.float64
        )

        # Definindo o espaço de observações (ph_t e conc_t)
        self.observation_space = spaces.Dict({
            "x1": spaces.Box(low=0, high=14, shape=(1,), dtype=np.float64), # ph
            "x2": spaces.Box(low=0, high=1000, shape=(1,), dtype=np.float64), # conc # TODO/Perguntar: validar min e max da concentração
            "y_ref": spaces.Box(low=0, high=14, shape=(1,), dtype=np.float64),
            "z_t": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float64)
        })
