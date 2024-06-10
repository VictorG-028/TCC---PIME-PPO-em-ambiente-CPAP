# from __future__ import annotations
from typing import Callable, Optional
from enums.ErrorFormula import ErrorFormula, error_functions
from enums.TerminationRule import TerminationRule, termination_functions
from modules.Scheduller import Scheduller
from modules.Ensemble import Ensemble

import numpy as np
import gymnasium
from gymnasium import spaces



class SetPointEnv(gymnasium.Env):
    """
    Um ambiente base para embrulhar formular que simulam um ambiente de set_point

    Parâmetros:
        scheduller (Scheduller): Controla o set_point ao longo dos steps
        params_ensemble (Ensemble): Controla os parâmetros aleatórios das simulações.
        simulation_model (Callable): Formula usada na step
        start_set_point (float): O ponto de ajuste inicial do ambiente.
    """

    metadata = {
        "render_modes": ["terminal"],
    }

    def __init__(
            self,
            scheduller: Scheduller,
            simulation_model: Callable,
            ensemble_params: dict[str, np.float64],
            termination_rule: TerminationRule | Callable = TerminationRule.INTERVALS,
            error_formula: ErrorFormula | Callable = ErrorFormula.DIFFERENCE_SQUARED,
            start_points: list[np.float64] = [0],
            tracked_point: str = 'x1',
            render_mode: str = "terminal",
            ):
        
        super(SetPointEnv, self).__init__()

        self.scheduller        = scheduller
        self.simulation_model  = simulation_model
        self.ensemble_params   = ensemble_params
        self.start_points      = start_points
        self.tracked_point     = tracked_point
        self.render_mode       = render_mode
        self.cummulative_error = 0
        
        print(f"[set_point_env.__init__()] var {termination_rule} of type {type(termination_rule)}")
        print(f"[set_point_env.__init__()] var {error_formula} of type {type(error_formula)}")
        # TODO use "import signarute from inspect" to check if termination_rule and error_formula have right signarute
        if callable(termination_rule):
            self.termination_rule = termination_rule
        elif isinstance(termination_rule, str):
            self.termination_rule = termination_functions[termination_rule]
        if callable(error_formula):
            self.error_formula    = error_formula
        elif isinstance(termination_rule, str):
            self.error_formula    = error_functions[error_formula]

        named_points = {
            f"x{i+1}": value for i, value in enumerate(self.start_points)
        }
        self.setpoint = self.scheduller.get_set_point_at(step=0)
        self.observation = {
            **named_points,
            "y_ref": self.setpoint,
            "z_t": 0
        }
        self.timestep: int = 0

        self.action_space = spaces.Box(low=-1_000, high=1_000, shape=(1,), dtype=np.float64)
        named_spaces = {
            f"x{i+1}": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64) 
            for i, _ in enumerate(self.start_points)
        }
        self.observation_space = spaces.Dict(
            {
                **named_spaces,
                "y_ref": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
                "z_t": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
            }
        )


    def set_ensemble_params(self, ensemble_params: dict[str, np.float64]):
        self.ensemble_params = ensemble_params


    def reset(self, seed: Optional[int] = None, options = None):
        super().reset(seed=seed)

        self.setpoint = self.scheduller.get_set_point_at(step=0)
        self.cummulative_error = 0
        # TODO: inicializar (x_1, ..., x_n)de forma aleatória em vez de start points
        self.observation = {
            **{f"x{i+1}": value for i, value in enumerate(self.start_points)},
            "y_ref": self.setpoint,
            "z_t": 0
        }
        self.reward = 0
        self.done = False
        self.truncated = {}

        self.timestep = 0

        return self.observation, {}
    

    def step(self, action: np.float64):
        set_point: np.float64 = self.scheduller.get_set_point_at(step=self.timestep)
        print(f"debugging observation (x_vector)")
        print(list(self.observation.values())[0:-1])
        x_vector: np.float64 = self.simulation_model(
            action, 
            *list(self.observation.values())[0:-2], # Get only x1 .. xn values; don't use y_ref and z_t (last 2 values)
            **self.ensemble_params
        )

        self.reward = self.error_formula(x_vector[self.tracked_point.value], set_point)
        self.cummulative_error += self.reward
        self.observation = {
            **x_vector,
            "y_ref": set_point,
            "z_t": self.cummulative_error
        }

        self.done = self.termination_rule(self.timestep, self.scheduller.intervals_sum)

        return self.observation, self.reward, self.done, {}
    
