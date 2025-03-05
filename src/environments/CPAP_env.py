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


# Currently not used
# class SimulationState(TypedDict):
#     i: int # 0,1,2,3... nunca zera e serve de índice para acessar arrays
#     phase_counter: int # Contador que zera ao trocar phase
#     phase: Literal["exhale", "inhale", "pause"]
#     start_phase_time: float



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
            x_size: int = 3,
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
            max_step: int = 30000, # simulation_time [s]
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
            "x1": spaces.Box(low=-1700, high=350, shape=(1,), dtype=np.float64),  # flow [ml /s] (valores low/high extremo considerando fluxo de ar negativo/positivo durante expiração forçada)
            "x2": spaces.Box(low=0, high=400, shape=(1,), dtype=np.float64),     # volume [ml] (400 ml é a quantidade de ar inserido pelo CPAP)
            "x3": spaces.Box(low=0, high=60, shape=(1,), dtype=np.float64),    # pressure [cmH2O] (-20 é um valor extremo durante expiração e 60 é um valor extremo durante ventilação mecânica)
            "x4": spaces.Box(low=-20, high=60, shape=(1,), dtype=np.float64),  # last pressure [cmH2O]
            "x5": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float64),     # Flag de inspiração [0 ou 1]
            "x6": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float64),     # Flag de expiração [0 ou 1]
            "x7": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float64),     # Flag de pausa [0 ou 1]
            "x8": self.action_space,                                           # last action taken [-1 até +1]
            "y_ref": spaces.Box(low=0, high=60, shape=(1,), dtype=np.float64),   # set point pressure [cmH2O] (60 é valor extremo durante ventilação mecânica)
            "z_t": spaces.Box(low=0, high=25, shape=(1,), dtype=np.float64)      # Acumulador de erro [cmH2O]
        })

    
    # Wrapper x_vector inicialization to apply a especial rule to this environment
    def reset(self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None) -> tuple[dict[str, Any], dict]:
        obs, _ = super().reset(seed, options)

        obs["x1"] = 0
        obs["x2"] = 0
        obs["x3"] = 0

        is_positive = obs["x1"] > 0
        is_negative = obs["x1"] < 0
        is_zero = obs["x1"] == 0

        obs["x4"] = 0 # Consider last pressure as 0
        obs["x5"] = int(is_positive)
        obs["x6"] = int(is_negative)
        obs["x7"] = int(is_zero)
        obs["x8"] = 0 # Consider last action as no volt applyed

        map_x1_to_phase = {
            (True, False, False): "inhale",
            (False, True, False): "exhale",
            (False, False, True): "pause"
        }

        # Reset accordinly to x vector private state of simulation model
        self.error = self.error_formula(y_ref=self.setpoint, y=obs[self.tracked_point])
        self.reward = self.reward_formula(y_ref=self.setpoint, y=obs[self.tracked_point])
        self.i = 0
        self.phase_counter = 1
        self.phase = "exhale" # map_x1_to_phase[(is_positive, is_negative, is_zero)]
        self.start_phase_time = 0
        self.last_pause_pressure = 5 # deve ser igual ao peep quando não existe a última pressão da pausa
        self.stop = False
        self.control_filtered = 0

        # print(f"reset foi invocado e resultou na {obs=}")
        # input(">>>")
        return obs, _

    
    class Lung:
        def __init__(self, r_aw=3, c_rs=60):
            """
            r_aw [cmH2O / l / s] \\
            _r_aw [cmH2O / ml / s] \\
            c_rs [ml / cmH2O]
            """
            # Normal range: 2 to 5 cmH2O/L/s
            # Airway resistance [cmH2O / l / s]
            self.r_aw = r_aw

            # Converted airway resistance = _r_aw = r_aw [cmH2O / l / s] / [1000 ml / l] = [cmH2O / ml / s]
            self._r_aw = r_aw / 1000

            # Normal range: 85 to 100 ml/cmH2O
            # Respiratory system compliance [ml / cmH2O]
            self.c_rs = c_rs

            self._rc = self._r_aw * c_rs


    class Ventilator:
        def __init__(self, v_t=350, u_t = 0, kb=1.0, tb=1.0, dt=1/30, rr=15, t_i=1, t_ip=0.25):
            """
            v_t [ml] \\
            peep [cmH2O] \\
            rr [min^(-1)] \\
            _rr [Hz] \\
            t_i [s] \\
            t_ip [s] \\
            t_c [s] \\
            t_e [s] \\
            _f_i [ml / s]
            """

            # Tidal volume [ml]
            self.v_t = v_t
            
            # Positive End Expiratory Pressure [cmH2O]
            self.peep = kb * u_t * (1 - np.exp(-dt / tb))

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
            self.t_e = self.t_c - self.t_i - self.t_ip

            # Inspiratory flow [ml / s]
            self._f_i = self.v_t / (self.t_i)


    def simulation_model(self,
                         u_t,                                            # action
                         current_flow, current_volume, current_pressure, # x vector
                         last_pressure, was_exhale, was_inhale, was_pause, last_action,
                         /, *,
                         r_aw, c_rs,                            # Lung       # Generated by ensemble
                         tv, kv, v_t, rr, t_i, t_ip,            # Ventilator # Generated by ensemble
                         dt, f_s,                               # Discretization constants
                        ) -> dict[str, float]:
        """ 
        u_t -> action [V] \\
        current_flow [ml / s], \\
        current_volume [ml], \\
        current_pressure [cmH2O] \\
        \\
        tv -> ventilator constant time [s] \\
        kv -> ventilator gain [ml / s / V] \\
        \\
        r_aw -> airway resistance [cmH2O / L / s] \\
        c_rs -> respiratory system compliance [ml / cmH2O] \\
        _rc -> _r_aw * c_rs
        \\
        v_t -> tidal volume [ml] \\
        peep -> positive end expiratory pressure [cmH2O] \\
        rr -> respiratory rate [min^(-1)] \\
        t_i -> inspiratory time [s] \\
        t_ip -> inspiratory pause time [s] \\
        \\
        dt -> sample time [s] \\
        f_s -> sample frequency [Hz]
        """
        exhale_sigma = 10.0
        inhale_sigma = 10.0
        pause_sigma = 10.0

        # Parâmetros do filtro
        tau_f = 0.1        # constante de tempo do filtro (em segundos)
        alpha = dt / tau_f # ganho do filtro
        # K_corr = 1.0       # Ganho de correção para converter a ação filtrada em pressão (cmH2O/V)


        current_time = self.i * dt

        lung = CpapEnv.Lung(r_aw, c_rs)
        ventilator = CpapEnv.Ventilator(v_t, self.setpoint+u_t, kv, tv, self.phase_counter * dt, rr, t_i, t_ip)
        
        # FIltro de passa baixa (low-pass)
        self.control_filtered = self.control_filtered + alpha * (ventilator.peep - self.control_filtered)

        # Converter a ação filtrada em um ajuste para o PEEP
        # ventilator.peep =  K_corr * self.control_filtered
        ventilator.peep = self.control_filtered

        last_pressure = current_pressure
        
        if self.phase == 'exhale':
            
            if self.phase_counter == 1:
                self.start_phase_time = current_time
                if (self.i == 0):
                    self.last_pause_pressure = ventilator.peep

            # if self.i == 4998:
            #     pass

            exhale_noise = np.random.normal(0, exhale_sigma)
            elapsed_phase_time = current_time - self.start_phase_time
            current_flow = (ventilator.peep - self.last_pause_pressure) / lung._r_aw * np.exp(-elapsed_phase_time / lung._rc) + exhale_noise
            # print(f"current_flow @@@ {current_time=} | {self.start_phase_time=} | {np.exp(-elapsed_phase_time / lung._rc)=} | {temp=} | {current_flow=}")
                        
            current_volume = current_volume + current_flow * dt
            
            current_pressure = current_flow * lung._r_aw + current_volume / lung.c_rs + ventilator.peep
            # self.last_pause_pressure = current_pressure
            # if (self.stop):
            #     self.stop = False
            print(f"action @@@ {u_t=} | {ventilator.peep=} | {self.last_pause_pressure=} | {elapsed_phase_time=}")

            self.phase_counter += 1
            if (self.phase_counter >= ventilator.t_e * f_s):
                self.phase = 'inhale'
                self.phase_counter = 1


        elif self.phase == 'inhale':
            
            pause_noise = np.random.normal(0, inhale_sigma)
            current_flow = ventilator._f_i + pause_noise

            if self.i > 0:
                current_volume = current_volume + current_flow * dt
            else:
                current_volume = 0

            current_pressure = current_flow * lung._r_aw + current_volume / lung.c_rs + ventilator.peep

            self.phase_counter += 1
            if (self.phase_counter >= ventilator.t_i * f_s):
                self.phase = 'pause'
                self.phase_counter = 1

        elif self.phase == 'pause':
            
            pause_noise = np.random.normal(0, pause_sigma)
            current_flow = pause_noise
            current_volume = current_volume + current_flow * dt
            current_pressure  = lung._r_aw * current_flow + current_volume / lung.c_rs + ventilator.peep
                
            self.phase_counter += 1
            if (self.phase_counter >= ventilator.t_ip * f_s):
                self.phase = 'exhale'
                self.phase_counter = 1
                self.last_pause_pressure = current_pressure
                # print(current_pressure)
                # self.stop = True

        self.i += 1
        # print(f"STEP @@@ {u_t=} {ventilator.peep=} {self.last_pause_pressure=} @@@ {self.i=} | {self.phase=} | {current_flow=} | {current_volume=} | {current_pressure=} | {ventilator.t_ip * f_s=} | {self.phase_counter=}")
        # input(">>>")

        # Lembrar: Modificar vetor x de simulation_model implica em modificar o x_size no create_cpap_environment, o observation space no __init__ e o reset desse env
        current_pressure = np.clip(current_pressure, -14, 14)
        return {
            "x1": current_flow,                   # Fluxo de ar atual
            "x2": current_volume,                 # Volume de ar atual
            "x3": current_pressure,               # Pressão de ar atual
            "x4": last_pressure,                # Pressão anterior a atual
            "x5": int(self.phase == 'inhale'),  # Flag de inspiração
            "x6": int(self.phase == 'exhale'),  # Flag de expiração
            "x7": int(self.phase == 'pause'),   # Flag de pausa
            "x8": u_t,                          # last action taken
        }
    

    @staticmethod
    def _expiratory_flow(lung, ventilator, last_pressure, start_time, current_time, blower_flow):
        """
        _t [s] \\
        _rc [s] = [cmH2O / ml / s] * [ml / cmH2O] \\
        np.exp(-_t / _rc) [Adimensional] \\
        passive_flow [ml / s] = [cmH2O] / [cmH2O / ml / s]
        """
        _t = current_time - start_time
        _rc = lung._r_aw * lung.c_rs
        temp = np.exp(-_t / _rc)
        if temp < 0.01:
            temp = 0
        passive_flow = ((blower_flow - last_pressure) / lung._r_aw) * temp
        print(f" _expiratoryFloy @@@ {current_time=} | {start_time=} | {np.exp(-_t / _rc)=} | {temp=} | {passive_flow=}")
        return passive_flow
    

    @staticmethod
    def create_cpap_environment(seed = 42,
                                set_points: list[float] =  [6, 9, 7],
                                intervals: list[float] = [5000, 5000, 5000],
                                distributions: dict[str, tuple[str, dict[str, float]]] = None,
                                integrator_bounds: tuple[float, float] = (-25, 25),
                                ppo_action_bounds: tuple[float, float] = (-1, 1),
                                ppo_observation_min_bounds: tuple[float, float] = (-np.inf, -np.inf, -14),
                                ppo_observation_max_bounds: tuple[float, float] = (np.inf, np.inf, 14),
                                pid_type: Literal["PID", "PI", "P"] = "PI",
                                max_step: int = 30_000,
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
                        x_size                 = 3+5,
                        x_start_points         = None,
                        tracked_point          = 'x3',
                        termination_rule       = TerminationRule.MAX_STEPS,
                        error_formula          = ErrorFormula.DIFFERENCE,
                        reward_formula         = RewardFormula.DIFFERENCE_SQUARED,
                        integrator_clip_bounds = integrator_bounds,
                        action_bounds          = ppo_action_bounds,
                        max_step               = max_step,  
                        )
        env = DictToArrayWrapper(env)

        # # Define model symbols
        # s = sp.symbols('s')
        # tb, kb = sp.symbols('tb kb')
        # rp, rl, c = sp.symbols('rp rl c')
        # # kp, ki, kd = sp.symbols('kp ki kd') # Not used

        # # Define model values
        # patient = {
        #     # hh: Heated Humidifier.
        #     # hme: Heat-and-moisture exchanger.
        #     'Heated Humidifier, Normal':             {'rp': 10e-3, 'c': 50, 'rl': 48.5 * 60 / 1000 },
        #     'Heated Humidifier, COPD':               {'rp': 20e-3, 'c': 60, 'rl': 48.5 * 60 / 1000 },
        #     'Heated Humidifier, mild ARDS':          {'rp': 10e-3, 'c': 45, 'rl': 48.5 * 60 / 1000 },
        #     'Heated Humidifier, moderate ARDS':      {'rp': 10e-3, 'c': 40, 'rl': 48.5 * 60 / 1000 },
        #     'Heated Humidifier, severe ARDS':        {'rp': 10e-3, 'c': 35, 'rl': 48.5 * 60 / 1000 },
        #     'Heat Moisture Exchange, Normal':        {'rp': 15e-3, 'c': 50, 'rl': 48.5 * 60 / 1000 },
        #     'Heat Moisture Exchange, COPD':          {'rp': 25e-3, 'c': 60, 'rl': 48.5 * 60 / 1000 },
        #     'Heat Moisture Exchange, mild ARDS':     {'rp': 15e-3, 'c': 45, 'rl': 48.5 * 60 / 1000 },
        #     'Heat Moisture Exchange, moderate ARDS': {'rp': 15e-3, 'c': 40, 'rl': 48.5 * 60 / 1000 },
        #     'Heat Moisture Exchange, severe ARDS':   {'rp': 15e-3, 'c': 35, 'rl': 48.5 * 60 / 1000 },
        # }
        # _rp, _c, _rl =  10e-3, 50, 48.5 * 60 / 1000
        # _tb = 10e-3
        # _kb = 0.5

        # # Define cpap model
        # blower_model = kb / (s + 1 / tb)
        # blower_model = sp.collect(blower_model, s)
        # patient_model = (rl + rp * rl * c * s) / (1 + (rp+ rl) * c * s)
        # patient_model = sp.collect(patient_model, s)
        # cpap_model = blower_model * patient_model
        # numerators, denominators = sp.fraction(cpap_model)
        # numerators = sp.Poly(numerators, s)
        # denominators = sp.Poly(denominators, s)
        # numerators = numerators.all_coeffs()  # Tranfer function numerator.
        # denominators = denominators.all_coeffs()  # Tranfer function denominator.

        # filled_numerators = list()
        # filled_denominators = list()
        # for numerator_coef, denominator_coef in zip(numerators, denominators):
        #     filled_numerators.append(numerator_coef.evalf(subs=dict(zip( (c, rp, tb, kb, rl), (_c, _rp, _tb, _kb, _rl) ))))
        #     filled_denominators.append(denominator_coef.evalf(subs=dict(zip( (c, rp, tb, kb, rl), (_c, _rp, _tb, _kb, _rl) ))))
        # filled_numerators = np.array(filled_numerators, dtype=np.float64)
        # filled_denominators = np.array(filled_denominators, dtype=np.float64)

        # cpap_model = ct.TransferFunction(filled_numerators, filled_denominators)

        # # Train the PID controller
        # trained_pid, pid_optimized_params = PIDController.train_pid_controller(
        #     cpap_model, 
        #     pid_training_method='BFGS',
        #     pid_type=pid_type,
        # )
        trained_pid, pid_optimized_params = None, None

        return env, scheduller, ensemble, trained_pid, pid_optimized_params
