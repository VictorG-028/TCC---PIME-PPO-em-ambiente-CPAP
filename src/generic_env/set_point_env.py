# from __future__ import annotations
from typing import Callable, Optional
from base.Enums import ErrorFormula, NamedPoint, TerminationRule
from base.Scheduller import Scheduller
from base.Ensemble import Ensemble

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
        "render_modes": ["rgb_array"],
    }

    def __init__(
            self,
            scheduller: Scheduller,
            params_ensemble: Ensemble,
            simulation_model: Callable,
            termination_rule: TerminationRule = TerminationRule.INTERVALS,
            error_formula: ErrorFormula = ErrorFormula.DIFFERENCE_SQUARED,
            start_points: list[np.float64] = [0],
            tracked_point: NamedPoint = NamedPoint.X1,
            render_mode: str = "rgb_array"
            ):
        
        super(SetPointEnv, self).__init__()

        self.scheduller       = scheduller
        self.params_ensemble  = params_ensemble
        self.simulation_model = simulation_model
        self.termination_rule = termination_rule
        self.error_formula    = error_formula
        self.start_points     = start_points
        self.tracked_point    = tracked_point
        self.render_mode      = render_mode

        named_points = {
            f"x{i+1}": value for i, value in enumerate(self.start_points)
        }
        self.setpoint = self.scheduller.get_set_point_at(step=0)
        self.observation = {
            **named_points,
            "y_ref": self.setpoint
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
                "y_ref": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64)
            }
        )


    def reset(self, seed: Optional[int] = None, options = None):
        super().reset(seed=seed)

        self.setpoint = self.scheduller.get_set_point_at(step=0)
        self.observation = {
            **{f"x{i+1}": value for i, value in enumerate(self.start_points)},
            "y_ref": self.setpoint
        }
        self.reward = 0
        self.done = False
        self.truncated = {}

        self.timestep = 0

        return self.observation, {}
    

    def step(self, action: np.float64):
        set_point: np.float64 = self.scheduller.get_set_point_at(step=self.timestep)
        params: dict[str, np.float64] = self.params_ensemble.get_param_set()
        print("debugging observation (x_vector)")
        print(list(self.observation.values())[0:-1])
        x_vector: np.float64 = self.simulation_model(
            action, 
            *list(self.observation.values())[0:-1], # Get only x1 .. xn values; don't use y_ref (last) value
            **params
        )

        self.observation = {
            **x_vector,
            "y_ref": set_point
        }

        self.reward = self.error_formula(x_vector[self.tracked_point.value], set_point)

        self.done = self.termination_rule(self.timestep, self.scheduller.intervals_sum)

        return self.observation, self.reward, self.done, {}
    
