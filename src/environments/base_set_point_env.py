# from __future__ import annotations
from typing import Any, Callable, Dict, Literal, Optional
from enums.ErrorFormula import ErrorFormula, error_functions
from enums.TerminationRule import TerminationRule, termination_functions
from modules.Scheduller import Scheduller
from modules.EnsembleGenerator import EnsembleGenerator

import functools
import inspect

import numpy as np
import gymnasium
from gymnasium import spaces

################################################################################

def debug_print_extra_info(func): 
    """
    Puts a string cotaning the method name before the print statement.
    """
    @functools.wraps(func) 
    def wrapper(*args, **kwargs): 
        frame = inspect.currentframe().f_back 

        local_env = frame.f_locals.get('env') 
        local_self = frame.f_locals.get('self') 

        # First time is the 'env' inside locals, second and onwards is the 'self'
        if local_env:
            class_name = local_env.unwrapped.__class__.__name__
        elif local_self:
            class_name = local_self.unwrapped.__class__.__name__
        else:
            raise Exception("[Decorator debug_print_extra_info error] Can't find 'env' or 'self' in locals")

        method_name = frame.f_code.co_name 
        print(f"[{class_name}.{method_name}] ", end="") 
        return func(*args, **kwargs) 
    return wrapper

################################################################################


class BaseSetPointEnv(gymnasium.Env):
    """
    [en] 
        Base environment to wrapper the simultaion formulas that simulate the changes to state (X_vector)
        Parameters:
            scheduller (Scheduller): Controla o set_point ao longo dos steps
            simulation_model (Callable): Formula usada na step
            ensemble_params (Ensemble): Controla os parâmetros aleatórios das simulações.
            x_size (int): Size of x vector inside observation
            start_points (opcional): Vetor x inicial. Caso null, então o vetor é gerado aleatoriamente.
            termination_rule (Callable): TODO
            error_formula (Callable): TODO
            tracked_point (str): "x{number}" like string that representes the x_vector tracked element
            extra_inputs (Dict[str, any]): Contém quaisquer variáveis ​​extras que entram no modelo de simulação. A key 'dt' geralmente é expressa em segundos.
            render_mode (Literal["terminal"]): Onlçy supports terminal rendering.
    
    [pt-br] 
        Um ambiente base para encapsular formulas que simulam um ambiente set_point.

        Parameters:
            scheduller (Scheduller): Controla o set_point ao longo dos steps
            simulation_model (Callable): Formula usada na step
            ensemble_params (Ensemble): Controla os parâmetros aleatórios das simulações.
            x_size (int): Size of x vector inside observation
            start_points (opcional): Vetor x inicial. Caso null, então o vetor é gerado aleatoriamente.
            termination_rule (Callable): TODO
            error_formula (Callable): TODO
            tracked_point (str): cadeia de caracteres "x{número}" que representa o elemento rastreado do x_vector
            extra_inputs (Dict[str, any]): Holds any extra variables that goes into simulation model. dt key is usualy expressed in seconds.
            render_mode (Literal["terminal"]): Apenas suporta renderização no terminal.
    """

    metadata = {
        "render_modes": ["terminal"],
    }

    def __init__(
            self,
            scheduller: Scheduller,
            ensemble_params: dict[str, np.float64],
            termination_rule: TerminationRule | Callable = TerminationRule.INTERVALS,
            error_formula: ErrorFormula | Callable[[float, float], float] = ErrorFormula.DIFFERENCE_SQUARED,
            action_size: int = 1,
            x_size: int = 1,
            x_start_points: Optional[list[np.float64]] = None,
            tracked_point: str = 'x1',
            max_step: Optional[int] = None,
            render_mode: Literal["terminal"] = "terminal",
            should_define_simple_action_and_obs_spaces: bool = False,
            ):
        
        if (x_start_points):
            is_correct_size = x_size == len(x_start_points)
        else:
            is_correct_size = True

        assert x_size is not None and is_correct_size, \
            f"Must initialize x_size with valid integer. Recived '{x_size=}'."
        
        if (termination_rule == TerminationRule.MAX_STEPS):
            assert max_step is not None, \
                "max step can't be None when termination rule is MAX_STEPS."

        super(BaseSetPointEnv, self).__init__()

        self.scheduller        = scheduller
        self.ensemble_params   = ensemble_params
        self.action_size       = action_size
        self.x_size            = x_size
        self.x_start_points    = x_start_points
        self.tracked_point     = tracked_point
        self.max_step          = max_step
        self.render_mode       = render_mode

        self.cummulative_error: float = 0
        self.timestep: int            = 0
        self.setpoint: float          = self.scheduller.get_set_point_at(step=0)

        # print(f"[set_point_env.__init__()] var {termination_rule} of type {type(termination_rule)}")
        # print(f"[set_point_env.__init__()] var {error_formula} of type {type(error_formula)}")

        # TODO use "import signarute from inspect" to check if termination_rule and error_formula have right signarute
        if callable(termination_rule):
            self.termination_rule = termination_rule
        elif isinstance(termination_rule, TerminationRule):
            self.termination_rule = termination_functions[termination_rule]
        if callable(error_formula):
            self.error_formula    = error_formula
        elif isinstance(error_formula, ErrorFormula):
            self.error_formula    = error_functions[error_formula]

        
        assert x_size > 0, \
            "Size of x vector is invalid. The vector should be an array with lenght x_size (positive integer)."
        assert action_size > 0, \
            "Actions size is invalid. Should be positive integer. Agent must hgave at least 1 possible action."

        # Default: Doesn't define action and ovbservation space, since this is an base (abstract) env.
        # Can define a simple case if needed.
        if (should_define_simple_action_and_obs_spaces):
            self.action_space = spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(action_size,), 
                dtype=np.float64
            )
            named_spaces = {
                f"x{i+1}": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64) 
                for i in range(self.x_size)
            }
            self.observation_space = spaces.Dict(
                {
                    **named_spaces,
                    "y_ref": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
                    "z_t": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float64),
                }
            )


    def set_ensemble_params(self, sample_params: dict[str, np.float64]) -> None:
        self.ensemble_params = sample_params


    def generate_x_vector_randomly(self) -> dict[str, np.float64]:
        random_sample = self.observation_space.sample()

        # Remove os dois últimos valores (y_ref e z_t)
        random_sample.pop("y_ref", None)
        random_sample.pop("z_t", None)

        random_x_vector = {
            f"x{i+1}": random_value[0]
            for i, random_value in enumerate(random_sample.values())
        }

        return random_x_vector


    def reset(self, seed: Optional[int] = None, options = None) -> tuple[dict[str, Any], dict]:
        super().reset(seed=seed)

        self.set_ensemble_params(options["ensemble_sample_parameters"])

        self.setpoint = self.scheduller.get_set_point_at(step=0)
        self.cummulative_error = 0

        # Generate vector x values (x_1, ..., x_n)
        if self.x_start_points:
            x_values = { 
                f"x{i+1}": value for i, value in enumerate(self.x_start_points)
            }
        else:
            x_values = self. generate_x_vector_randomly() 

        self.observation = {
            **x_values,
            "y_ref": self.setpoint,
            "z_t": 0
        }
        self.reward = self.error_formula(x_values[self.tracked_point], self.setpoint)
        self.done = False
        self.truncated = {}

        self.timestep = 0

        return self.observation, {}
    

    @debug_print_extra_info
    def step(self, action: np.float64):
        set_point: np.float64 = self.scheduller.get_set_point_at(step=self.timestep)
        print(f"{action=}")
        # print(f"x_vector={list(self.observation.values())[0:-2]}")
        # print(f"y_ref={list(self.observation.values())[-2]}")
        # print(f"z_t={list(self.observation.values())[-1]}")
        # print(f"{self.ensemble_params=}")
        x_vector: dict[str, np.float64] = self.simulation_model(
            action, 
            *list(self.observation.values())[0:-2], # Get only x1 .. xn values; don't use y_ref and z_t (last 2 values)
            **self.ensemble_params
        )

        self.reward = self.error_formula(x_vector[self.tracked_point], set_point)
        self.cummulative_error += self.reward
        self.observation = {
            **x_vector,
            "y_ref": set_point,
            "z_t": self.cummulative_error,
        }

        self.timestep += 1 # Timestep should be updated before termination rule
        self.done = self.termination_rule(self.timestep, self.scheduller.intervals_sum, self.max_step)
        print(f"{self.done=} ({self.timestep}/{self.scheduller.intervals_sum})(timestep/scheduller.intervals_sum) ({self.timestep}/{self.max_step})(timestep/max_step)")

        # print(f"RETURNED OBS: {self.observation=}")
        return self.observation, self.reward, self.done, False, {}
    

    def simulation_model(self, u_t, *x_vector, **ensemble_params):
        raise NotImplementedError("Must crate a base class that defines simulation_model using BaseSetPointEnv as super class")
