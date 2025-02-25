from algorithms.PID_Controller import PIDController
from enums.RewardFormula import RewardFormula
from environments.base_set_point_env import BaseSetPointEnv
from typing import Any, Callable, Literal, Optional, TypedDict
from enums.ErrorFormula import ErrorFormula
from enums.TerminationRule import TerminationRule
from modules.Scheduller import Scheduller
from modules.EnsembleGenerator import EnsembleGenerator
from wrappers.DictToArray import DictToArrayWrapper

import numpy as np
import sympy as sp
import control as ct
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
            error_formula: ErrorFormula | Callable = ErrorFormula.DIFFERENCE,
            reward_formula: RewardFormula | Callable = RewardFormula.DIFFERENCE_SQUARED,
            start_points: Optional[list[np.float64]] = None,
            tracked_point: str = 'x3',
            integrator_clip_bounds: Optional[tuple[float, float]] = (-25, 25),
            render_mode: Literal["terminal"] = "terminal",

            # Created parameters
            action_bounds: tuple[float, float] = (0, 100), # [cmH2O]
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
            start_ensemble_params=ensemble_params,
            termination_rule=termination_rule,
            error_formula=error_formula,
            reward_formula=reward_formula,
            action_size=1,
            x_size=x_size,
            x_start_points=start_points,
            tracked_point=tracked_point,
            max_step=max_step,
            integrator_clip_bounds=integrator_clip_bounds,
            render_mode=render_mode,
        )

        # Definindo o espaço de ações (u_t)
        self.action_space = spaces.Box(
            low=action_bounds[0],
            high=action_bounds[1],
            dtype=np.float64
        )

        # Pergunta: O GPT-4o gerou números satisfatórios paro low e high do vetor x ?
        
        # Lembrar: Modificar vetor x aqui implica em modificar o retorno do simulation model e o reset do env
        # Definindo o espaço de observações (flow [l / min], volume [ml], pressure [cmH2O])
        self.observation_space = spaces.Dict({
            "x1": spaces.Box(low=-100, high=100, shape=(1,), dtype=np.float64),  # flow (valores low/high extremo considerando fluxo de ar negativo/positivo durante expiração forçada)
            "x2": spaces.Box(low=0, high=8000, shape=(1,), dtype=np.float64),    # volume (8000 ml é a capacidade pulmonar total máxima para adultos, considerando casos extremos)
            "x3": spaces.Box(low=-20, high=60, shape=(1,), dtype=np.float64),    # pressure (-20 é um valor extremo durante expiração e 60 é um valor extremo durante ventilação mecânica)
            # "x4": spaces.Box(low=-20, high=60, shape=(1,), dtype=np.float64),  # last pressure
            # "x5": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float64),     # Flag de inspiração
            # "x6": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float64),     # Flag de expiração
            # "x7": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float64),     # Flag de pausa
            # "x8": self.action_space,                                           # last action taken
            "y_ref": spaces.Box(low=0, high=60, shape=(1,), dtype=np.float64),   # set point pressure (60 é valor extremo durante ventilação mecânica)
            "z_t": spaces.Box(low=0, high=float('inf'), shape=(1,), dtype=np.float64)  # Acumulador de erro
        })

    
    # Wrapper x_vector inicialization to apply a especial rule to this environment
    def reset(self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None) -> tuple[dict[str, Any], dict]:
        obs, _ = super().reset(seed, options)

        is_positive = obs["x1"] > 0
        is_negative = obs["x1"] < 0
        is_zero = obs["x1"] == 0

        # obs["x4"] = 0 # Consider last pressure as 0
        # obs["x5"] = int(is_positive)
        # obs["x6"] = int(is_negative)
        # obs["x7"] = int(is_zero)
        # obs["x8"] = 0 # Consider last action as no volt applyed

        map_x1_to_phase = {
            (True, False, False): "inhale",
            (False, True, False): "exhale",
            (False, False, True): "pause"
        }

        # Reset accordinly to x vector Internal state of simulation model
        self.i = 0
        self.phase_counter = 1
        self.phase = map_x1_to_phase[(is_positive, is_negative, is_zero)]
        self.start_phase_time = 0

        return obs, _

    
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
            current_volume = current_volume + (current_flow * dt)
            current_pressure = lung._r_aw * current_flow + current_volume / lung.c_rs + ventilator.peep # P = F x R  +  V x E  +  PEEP
                
            self.phase_counter += 1
            if (self.phase_counter >= ventilator.t_ip * f_s):
                self.phase = 'exhale'
                self.phase_counter = 1
        
        current_flow = current_flow * 60 / 1000 # Converte [l / min] para [ml / s]

        self.i += 1

        # Lembrar: Modificar vetor x aqui implica em modificar o observation space no __init__ e o reset desse env
        return {
            "x1": current_flow,                   # Fluxo de ar atual
            "x2": current_volume,                 # Volume de ar atual
            "x3": current_pressure,               # Pressão de ar atual
            # "x4": last_pressure,                # Pressão anterior a atual
            # "x5": int(self.phase == 'inhale'),  # Flag de inspiração
            # "x6": int(self.phase == 'exhale'),  # Flag de expiração
            # "x7": int(self.phase == 'pause'),   # Flag de pausa
            # "x8": u_t,                          # last action taken
        }
    

    @staticmethod
    def _expiratory_flow(lung, ventilator, last_pressuse, start_time, current_time):
        _t = current_time - start_time
        _rc = lung._r_aw * lung.c_rs
        return (ventilator.peep - last_pressuse) / lung._r_aw * np.exp(-_t / _rc)
    

    @staticmethod
    def create_cpap_environment(seed = 42,
                                set_points: list[float] =  [5, 15, 10],
                                intervals: list[float] = [500, 500, 500],
                                distributions: dict[str, tuple[str, dict[str, float]]] = None,
                                integrator_bounds: tuple[float, float] = (-25, 25),
                                action_bounds: tuple[float, float] = (0, 100),
                                pid_type: Literal["PID", "PI", "P"] = "PI",
                                ) -> tuple[BaseSetPointEnv, Scheduller, EnsembleGenerator, Callable]:
        """ ## Variable Glossary

        s
            Variável complexa de Laplace. 
            Usada para representar a frequência em análises de sistemas no domínio de Laplace.

        ### Pacient variables
        rp
            Unidade de medida: [cmH2O/ml/s]
            Inspiratory Resistance. Resistência Inspiratória.
            Representa a resistência ao fluxo de ar durante a inspiração.
            Originally [cmH2O/L/s].
        rl
            Unidade de medida: [cmH2O/ml/s]
            Leak Resistance. Resistência a vazamentos.
            Representa a resistência ao fluxo de ar devido a vazamentos no sistema.
            Originally 48.5 [cmH2O/L/min].
            Intentional leakage resistance from Philips Respironics, model Amara Gel at 30 L/min. 
        c
            Unidade de medida: [ml/cmH2O]
            Static Compliance. Complacência estática.
            Representa a capacidade do pulmão de se expandir e contrair em resposta a mudanças de pressão.

        ### Blower variables
        tb
            Unidade de medida: [s]
            Blower constant time. Constante de tempo do soprador.
            Representa o tempo necessário para o soprador atingir uma fração significativa de sua resposta final.
        kb
            Unidade de medida: [cm³/s/V]
            V = voltagem aplicada ao soprador.
            cm³/s = fluxo volumétrico
            Blower Gain. Ganho do sporador.

        ### PID variables
        kp
            Unidade de medida: adimensional
            O ganho proporcional ajusta a contribuição proporcional ao erro no sinal de controle.
        ki 
            Unidade de medida: [1/s]
            O ganho integral ajusta a contribuição proporcional à integral do erro ao longo do tempo.
        kd
            Unidade de medida: [s] 
            O ganho derivativo ajusta a contribuição proporcional à taxa de variação do erro. 
        """

        scheduller = Scheduller(set_points, intervals)
        ensemble = EnsembleGenerator(distributions, seed)
        
        env = gymnasium.make("CpapEnv-V0", 
                        scheduller             = scheduller,
                        ensemble_params        = ensemble.generate_sample(),
                        x_size                 = 3,
                        x_start_points         = None,
                        tracked_point          = 'x3',
                        termination_rule       = TerminationRule.MAX_STEPS,
                        error_formula          = ErrorFormula.DIFFERENCE,
                        reward_formula         = RewardFormula.DIFFERENCE_SQUARED,
                        integrator_clip_bounds = integrator_bounds,
                        action_bounds          = action_bounds,
                        )
        env = DictToArrayWrapper(env)

        # Define model symbols
        s = sp.symbols('s')
        tb, kb = sp.symbols('tb kb')
        rp, rl, c = sp.symbols('rp rl c')
        # kp, ki, kd = sp.symbols('kp ki kd') # Not used

        # Define model values
        patient = {
            # hh: Heated Humidifier.
            # hme: Heat-and-moisture exchanger.
            'Heated Humidifier, Normal':             {'rp': 10e-3, 'c': 50, 'rl': 48.5 * 60 / 1000 },
            'Heated Humidifier, COPD':               {'rp': 20e-3, 'c': 60, 'rl': 48.5 * 60 / 1000 },
            'Heated Humidifier, mild ARDS':          {'rp': 10e-3, 'c': 45, 'rl': 48.5 * 60 / 1000 },
            'Heated Humidifier, moderate ARDS':      {'rp': 10e-3, 'c': 40, 'rl': 48.5 * 60 / 1000 },
            'Heated Humidifier, severe ARDS':        {'rp': 10e-3, 'c': 35, 'rl': 48.5 * 60 / 1000 },
            'Heat Moisture Exchange, Normal':        {'rp': 15e-3, 'c': 50, 'rl': 48.5 * 60 / 1000 },
            'Heat Moisture Exchange, COPD':          {'rp': 25e-3, 'c': 60, 'rl': 48.5 * 60 / 1000 },
            'Heat Moisture Exchange, mild ARDS':     {'rp': 15e-3, 'c': 45, 'rl': 48.5 * 60 / 1000 },
            'Heat Moisture Exchange, moderate ARDS': {'rp': 15e-3, 'c': 40, 'rl': 48.5 * 60 / 1000 },
            'Heat Moisture Exchange, severe ARDS':   {'rp': 15e-3, 'c': 35, 'rl': 48.5 * 60 / 1000 },
        }
        _rp, _c, _rl =  10e-3, 50, 48.5 * 60 / 1000
        _tb = 10e-3
        _kb = 0.5

        # Define cpap model
        blower_model = kb / (s + 1 / tb)
        blower_model = sp.collect(blower_model, s)
        patient_model = (rl + rp * rl * c * s) / (1 + (rp+ rl) * c * s)
        patient_model = sp.collect(patient_model, s)
        cpap_model = blower_model * patient_model
        numerators, denominators = sp.fraction(cpap_model)
        numerators = sp.Poly(numerators, s)
        denominators = sp.Poly(denominators, s)
        numerators = numerators.all_coeffs()  # Tranfer function numerator.
        denominators = denominators.all_coeffs()  # Tranfer function denominator.

        filled_numerators = list()
        filled_denominators = list()
        for numerator_coef, denominator_coef in zip(numerators, denominators):
            filled_numerators.append(numerator_coef.evalf(subs=dict(zip( (c, rp, tb, kb, rl), (_c, _rp, _tb, _kb, _rl) ))))
            filled_denominators.append(denominator_coef.evalf(subs=dict(zip( (c, rp, tb, kb, rl), (_c, _rp, _tb, _kb, _rl) ))))
        filled_numerators = np.array(filled_numerators, dtype=np.float64)
        filled_denominators = np.array(filled_denominators, dtype=np.float64)

        cpap_model = ct.TransferFunction(filled_numerators, filled_denominators)

        # Train the PID controller
        trained_pid, pid_optimized_params = PIDController.train_pid_controller(
            cpap_model, 
            pid_training_method='BFGS',
            pid_type=pid_type,
        )

        return env, scheduller, ensemble, trained_pid, pid_optimized_params
