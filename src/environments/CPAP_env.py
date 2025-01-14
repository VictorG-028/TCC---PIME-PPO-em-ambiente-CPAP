from environments.base_set_point_env import BaseSetPointEnv
from typing import Callable, Dict, Literal, Optional, TypedDict
from enums.ErrorFormula import ErrorFormula, error_functions
from enums.TerminationRule import TerminationRule, termination_functions
from modules.Scheduller import Scheduller
from modules.EnsembleGenerator import EnsembleGenerator

import numpy as np
import gymnasium
from gymnasium import spaces


class Lung:
    def __init__(self, r_aw=3, c_rs=60):
        # Normal range: 2 to 5 cmH2O/L/s
        # Airway resistance [cmH2O / l / s]
        self.r_aw = r_aw

        # Converted airway resistance = _r_aw = r_aw [cmH2O / l / s] / [1000 ml / l] = [cmH2O / ml / s]
        self._r_aw = r_aw / 1000

        # Normal range: 85 to 100 ml/cmH2O
        # Respiratory system compliance [ml / cmH2O]
        self.c_rs = c_rs


class Ventilator:
    def __init__(self, v_t=350, peep=5, rr=15, t_i=1, t_ip=0.25):

        # Tidal volume [ml]
        self.v_t = v_t
        
        # Positive End Expiratory Pressure [cmH2O]
        self.peep = peep

        # Normal range: 10 to 20 min^(-1)
        # Respiratory rate [min^(-1)]
        self.rr = rr

        # Converted Respiratory rate = _rr = rr * [min / 60 s] = rr / 60 [s^(-1)] = rr / 60 [Hz]
        self._rr = rr / 60

        # Inspiratory Time [s]
        self.t_i = t_i

        # Inspiratory pause time [s]
        self.t_ip = t_ip
        
        # Cicle time = 1 / _rr [s]
        self.t_c = 1 / self._rr

        # Expiratory Time [s]
        self.t_e = self.t_c - self.t_i  - self.t_ip

        # Inspiratory flow [ml / s]
        self._f_i = self.v_t / (self.t_i)


class SimulationState(TypedDict):
    i: int # 0,1,2,3... nunca zera e serve de índice para acessar arrays
    phase_counter: int # Contador que zera ao trocar phase
    phase: Literal["exhale", "inhale", "pause"]
    start_phase_time: float



class CpapEnv(BaseSetPointEnv):
    """
    Environment specific for Cascade Water Tank.
    This class defines state and action spaces.
    """
    
    # simulation model variables
    i: int                                      = 0 # 0,1,2,3... nunca zera e serve de índice para acessar arrays
    phase_counter: int                          = 1 # Contador que zera ao trocar phase
    phase: Literal["exhale", "inhale", "pause"] = "exhale"
    start_phase_time: float                     = 0

    def __init__(
            self,

            # Base class parameters
            scheduller: Scheduller,
            ensemble_params: dict[str, np.float64],
            x_size: int = 2,
            x_start_points: Optional[list[np.float64]] = None,
            termination_rule: TerminationRule | Callable = TerminationRule.MAX_STEPS,
            error_formula: ErrorFormula | Callable = ErrorFormula.DIFFERENCE_SQUARED,
            start_points: Optional[list[np.float64]] = None,
            tracked_point: str = 'x3',
            render_mode: Literal["terminal"] = "terminal",

            # Created parameters
            max_step: int = 30, # simulation_time [s]
            sample_frequency: int = 1000, # [Hz]
            ):
        
        # print(f"{ensemble_params=}")
        # print(f"{ensemble_params['peep']=}")
        
        # assert isinstance(lung, Lung), \
        #     "lung model must not be None. Should pass a Lung class instance."
        # assert isinstance(ventilator, Ventilator), \
        #     "lung model must not be None. Should pass a Lung class instance."
        
        # assert action_size == 1, \
        #     "action_size must be equal to 1."
        

        self.max_step = max_step       # [s] = 1/sample_frequency

        super().__init__(
            scheduller=scheduller,
            ensemble_params=ensemble_params,
            termination_rule=termination_rule,
            error_formula=error_formula,
            action_size=1,
            x_size=x_size,
            x_start_points=start_points,
            tracked_point=tracked_point,
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

    
    def simulation_model(self,
                         u_t,                                            # action
                         current_flow, current_volume, current_pressure, # x vector
                         /, *, 
                         rp, c, rl,                # Pacient    # Generated by ensemble
                         tb, kb,                   # Blower     # Generated by ensemble
                         r_aw, c_rs,               # Lung       # Generated by ensemble
                         v_t, peep, rr, t_i, t_ip, # Ventilator # Generated by ensemble
                         dt, f_s,                               # Constants generated by ensemble
                        ) -> dict[str, float]:
        last_pressure = current_pressure

        lung = Lung(r_aw, c_rs)
        ventilator = Ventilator(v_t, peep, rr, t_i, t_ip)
        

        if self.phase == 'exhale':
            current_time = self.i * dt

            if self.phase_counter == 0:
                self.start_phase_time = current_time

            current_flow = CpapEnv._expiratory_flow(lung, 
                                                    ventilator, 
                                                    current_pressure, 
                                                    self.start_phase_time, 
                                                    current_time
                                                    )
                        
            current_volume = current_volume + current_flow * dt

            current_pressure = current_flow * lung._r_aw + current_volume / lung.c_rs + ventilator.peep

            self.phase_counter += 1
            if (self.phase_counter >= ventilator.t_e * f_s):
                self.phase = 'inhale'
                self.phase_counter = 1


        elif self.phase == 'inhale':

            current_flow = ventilator._f_i

            if self.i > 0:
                current_volume = current_volume + (current_flow * dt)
            else:
                current_volume = 0

            current_pressure = current_flow * lung._r_aw + current_volume / lung.c_rs + ventilator.peep

            self.phase_counter += 1
            if (self.phase_counter >= ventilator.t_i * f_s):
                self.phase = 'pause'
                self.phase_counter = 1

        elif self.phase == 'pause':

            current_flow = 0
            current_volume  = current_volume + (current_flow * dt)
            current_pressure  = lung._r_aw * current_flow + current_volume / lung.c_rs + ventilator.peep # P = F x R  +  V x E  +  PEEP
                
            self.phase_counter += 1
            if (self.phase_counter >= ventilator.t_ip * f_s):
                self.phase = 'exhale'
                self.phase_counter = 1
        
        current_flow = current_flow * 60 / 1000 # Converte [l / min] para [ml / s]

        self.i += 1

        return {
            "x1": current_flow,    # Fluxo de ar atual
            "x2": current_volume,  # Volume de ar atual
            "x3": current_pressure,# Pressão de ar atual
            # "x4": last_pressure,   # Pressão anterior a atual
            # "x5": self.phase == 'inhale',  # Flag de inspiração
            # "x6": self.phase == 'exhale',  # Flag de expiração
            # "x7": self.phase == 'pause'    # Flag de pausa
        }
    

    @staticmethod
    def _expiratory_flow(lung, ventilator, last_pressuse, start_time, current_time):
        _t = current_time - start_time
        _rc = lung._r_aw * lung.c_rs
        return (ventilator.peep - last_pressuse) / lung._r_aw * np.exp(-_t / _rc)
        
