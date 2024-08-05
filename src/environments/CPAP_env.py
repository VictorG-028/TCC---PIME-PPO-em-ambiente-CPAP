from environments.base_set_point_env import BaseSetPointEnv
from typing import Callable, Dict, Literal, Optional
from enums.ErrorFormula import ErrorFormula, error_functions
from enums.TerminationRule import TerminationRule, termination_functions
from modules.Scheduller import Scheduller
from modules.Ensemble import Ensemble

import numpy as np
import gymnasium
from gymnasium import spaces


class Lung:
    def __init__(self, r_aw=3, c_rs=60):
        # Airway resistance
        # Normal range: 2 to 5 cmH2O/L/s
        # r_aw [cmH2O / l / s]
        self.r_aw = r_aw

        # _r_aw = r_aw [cmH2O / l / s] / [1000 ml / l] = [cmH2O / ml / s]
        self._r_aw = r_aw / 1000

        # Respiratory system compliance
        # Normal range: 85 to 100 ml/cmH2O
        # c_rs [ml / cmH2O]
        self.c_rs = c_rs


class Ventilator:
    def __init__(self, v_t=350, peep=5, rr=15, t_i=1, t_ip=0.25):
        # Tidal volume
        # v_t [ml]
        self.v_t = v_t
        
        # Positive End Expiratory Pressure
        # peep [cmH2O]
        self.peep = peep

        # Respiratory rate
        # Normal range: 10 to 20 min^(-1)
        # rr [min^(-1)]
        self.rr = rr
        # _rr = rr * [min / 60 s] = rr / 60 [s^(-1)] = rr / 60 [Hz]
        self._rr = rr / 60

        # Inspiratory Time
        # t_i [s]
        self.t_i = t_i

        # Inspiratory pause time
        # t_pi [s]
        self.t_ip = t_ip
        
        # Cicle time
        # t_c = 1 / _rr [s]
        self.t_c = 1 / self._rr

        # Expiratory Time
        # t_e [s]
        self.t_e = self.t_c - self.t_i  - self.t_ip

        # Inspiratory flow
        # _f_i [ml / s]
        self._f_i = self.v_t / (self.t_i)


class SimulationState:
    def __init__(self, state_dict: Dict[str, any]) -> None:
        self.state_dict = state_dict


class CpapEnv(BaseSetPointEnv):
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
            x_size: int = 2,
            x_start_points: Optional[list[np.float64]] = None,
            termination_rule: TerminationRule | Callable = TerminationRule.MAX_STEPS,
            error_formula: ErrorFormula | Callable = ErrorFormula.DIFFERENCE_SQUARED,
            start_points: Optional[list[np.float64]] = None,
            tracked_point: str = 'x3',
            render_mode: Literal["terminal"] = "terminal",

            # Created parameters
            max_step: int = 30, # simulation_time
            sample_frequency: int = 1000,
            ):
        
        # print(f"{ensemble_params=}")
        # print(f"{ensemble_params['peep']=}")
        
        # assert isinstance(lung, Lung), \
        #     "lung model must not be None. Should pass a Lung class instance."
        # assert isinstance(ventilator, Ventilator), \
        #     "lung model must not be None. Should pass a Lung class instance."
        
        # assert action_size == 1, \
        #     "action_size must be equal to 1."
        
        # [Simulation.__init__]
        # self.lung = lung
        # self.ventilator = ventilator
        self.f_s = sample_frequency  # [Hz]
        self.dt = 1 / sample_frequency  # [s]
        self.max_step = max_step  # [s]

        # [Simulation.simulate]
        self.t = np.arange(start=0, stop=self.max_step, step=self.dt)  # Time [s]
        self.p = np.zeros(len(self.t))  # Pressure [cmH2O]
        self._f = np.zeros(len(self.t))  # Flow [ml / s]
        self.f = np.zeros(len(self.t))  # Flow [l / min]
        self.v = np.zeros(len(self.t))  # Volume [ml]

        # Perguntas sobre initial state
        # Pergunta: onde driving_pressure é usado na simulação ?
        # Pergunta: onde drive last_pressure é usado na simulação ?

        # [Simulation.simulate] Initial state
        self.phase = 'exhale'
        self.last_pressuse = ensemble_params['peep'] # TODO: fazer esse valor mudar ao trocar o ensemble. provavelmente mudar o reset do BaseEnv
        self._f[0] = 0
        self.v[0] = 0
        self.phase_counter = 0
        self.start_phase_time = 0
        self.driving_pressure = 0
        self.i = 0

        # Extra inputs
        extra_inputs = {
            'dt': 1 / sample_frequency, 
            'f_s': self.f_s,
            # 'lung': lung,  # Faz parte do ensemble
            # 'ventilator': ventilator, # Faz parte do ensemble
            't': self.t, 
            'p': self.p, 
            '_f': self._f, 
            'f': self.f,
            'v': self.v,
            # TODO: testar pra ver se o estado persiste fora de SimulationState
            'phase': self.phase,
            # 'last_pressuse': self.last_pressuse, # Faz parte do vetor x
            'phase_counter': self.phase_counter,
            'start_phase_time': self.start_phase_time,
            # 'driving_pressure': self.driving_pressure,
            'i': self.i,
            'statefull_obj': SimulationState({ # Keep state changes inside other functions that uses this obj
            })
        }

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
            extra_inputs=extra_inputs,
            max_step=max_step,
            render_mode=render_mode,
        )

        # Definindo o espaço de ações (u_t)
        self.action_space = spaces.Box(
            low=0,
            high=100,
            dtype=np.float64
        )

        # Pergunta: O GPT-4o gerou números satisfatórios paro low e high do vetor x ?

        # Definindo o espaço de observações (flow [l / min], volume [ml], pressure [cmH2O])
        self.observation_space = spaces.Dict({
            "x1": spaces.Box(low=-100, high=100, shape=(1,), dtype=np.float64),  # flow (valores low/high extremo considerando fluxo de ar negativo/positivo durante expiração forçada)
            "x2": spaces.Box(low=0, high=8000, shape=(1,), dtype=np.float64),    # volume (8000 ml é a capacidade pulmonar total máxima para adultos, considerando casos extremos)
            "x3": spaces.Box(low=-20, high=60, shape=(1,), dtype=np.float64),    # pressure (-20 é um valor extremo durante expiração e 60 é um valor extremo durante ventilação mecânica)
            "y_ref": spaces.Box(low=0, high=60, shape=(1,), dtype=np.float64),   # set point pressure (60 é valor extremo durante ventilação mecânica)
            "z_t": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float64)  # Acumulador de erro
        })
