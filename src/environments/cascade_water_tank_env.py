from environments.base_set_point_env import BaseSetPointEnv
from typing import Callable, Literal, Optional
from enums.ErrorFormula import ErrorFormula, error_functions
from enums.TerminationRule import TerminationRule, termination_functions
from modules.Scheduller import Scheduller
from modules.Ensemble import Ensemble

import numpy as np
import gymnasium
from gymnasium import spaces

class CascadeWaterTankEnv(BaseSetPointEnv):
    """
    Environment specific for Cascade Water Tank.
    This class defines state and action spaces.
    """

    def __init__(
            self,

            # Base class parameters
            scheduller: Scheduller,
            simulation_model: Callable,
            ensemble_params: dict[str, np.float64],
            action_size: int = 1,
            x_size: int = 2,
            x_start_points: Optional[list[np.float64]] = None,
            termination_rule: TerminationRule | Callable = TerminationRule.INTERVALS,
            error_formula: ErrorFormula | Callable = ErrorFormula.DIFFERENCE_SQUARED,
            start_points: Optional[list[np.float64]] = None,
            tracked_point: str = 'x2',
            render_mode: Literal["terminal"] = "terminal",

            # Created parameters
            observation_max_boundaries: list[float] = [100, 100], # TODO: ver se deve clipar a observação com base nas boudaries 
            ):
        
        # TODO criar wrapper que verifica se esse ambiente foi criado corretamente
        # assert action_size == 1 and x_size == 2, ""
        
        # ACTION_BOUNDARIES: list[tuple[float, float]] = [
        #     (-1_000, 1_000) # First action (u_t) 
        # ]

        assert start_points is None or x_size == len(start_points), \
            "Lenght of start_points must be equal to x_size."
        assert x_size == len(observation_max_boundaries), \
            "Lenght of observation_max_boundaries must be equal to x_size."
        assert action_size == 1, \
            "action_size must be equal to 1."
        # assert action_size == len(ACTION_BOUNDARIES), \
        #     "Lenght of action_boundaries must be equal to action_size."
        
        super().__init__(
            scheduller=scheduller,
            simulation_model=simulation_model,
            ensemble_params=ensemble_params,
            termination_rule=termination_rule,
            error_formula=error_formula,
            action_size=1,
            x_size=x_size,
            x_start_points=start_points,
            tracked_point=tracked_point,
            extra_inputs={'dt': 2},
            render_mode=render_mode,
        )
        
        # Definindo o espaço de ações (u_t)
        self.action_space = spaces.Box(
            low= -1_000, 
            high= 1_000, 
            shape=(action_size,), 
            dtype=np.float64
        )

        # Explicit show the default case
        if (x_size == 2): # Default case
            x_vector = {
                "x1": spaces.Box(low=0, high=observation_max_boundaries[0], shape=(1,), dtype=np.float64), # Nível da água do primeiro tanque
                "x2": spaces.Box(low=0, high=observation_max_boundaries[1], shape=(1,), dtype=np.float64), # Nível da água do segundo tanque
            }
        else: # General case
            x_vector = {
                f"x{i+1}": spaces.Box(low=0, high=observation_max_boundaries[i], shape=(1,), dtype=np.float64) 
                for i in range(x_size)
            }

        # Definindo o espaço de observações (l1_t e l2_t)
        self.observation_space = spaces.Dict({
            **x_vector,
            "y_ref": spaces.Box(low=0, high=observation_max_boundaries[1], shape=(1,), dtype=np.float64),
            "z_t": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float64)
        })

        
