from algorithms.PID_Controller import PIDController
from enums.RewardFormula import RewardFormula
from environments.base_set_point_env import BaseSetPointEnv
from typing import Any, Callable, Literal, Optional
from enums.ErrorFormula import ErrorFormula, error_functions
from enums.TerminationRule import TerminationRule, termination_functions
from modules.Scheduller import Scheduller
from modules.EnsembleGenerator import EnsembleGenerator
from wrappers.DictToArray import DictToArrayWrapper

import numpy as np
import sympy as sp
import control as ct
import gymnasium
from gymnasium import spaces

# 1. Função de atualização do estado (modelo dinâmico do Double Water Tank)
def double_tank_update(t, x, u, params):
    """
    Atualiza o estado do tanque com dt = 2 (modelo discreto).
    """
    l1, l2 = x  # Estado atual 
    u_t = max(0, u[0]) # Entrada (ação da bomba)

    # Obtém parâmetros do sistema
    g = params.get("g", 981)        # Gravidade [cm/s²]
    p1 = params.get("p1", 0.0020)   # Razão área do tanque superior
    p2 = params.get("p2", 0.0020)   # Razão área do tanque inferior
    p3 = params.get("p3", 0.1)      # Razão da bomba
    dt = params.get("dt", 2)        # Tempo de amostragem

    # Discretização explícita (Euler)
    l1_next = l1 + dt * (-p1 * np.sqrt(2 * g * max(l1, 0)) + p3 * u_t)
    l2_next = l2 + dt * (p1 * np.sqrt(2 * g * max(l1, 0)) - p2 * np.sqrt(2 * g * max(l2, 0)))

    # Aplicar restrições (0 ≤ l1_t, l2_t ≤ 10)
    l1_next = np.clip(l1_next, 0, 10)
    l2_next = np.clip(l2_next, 0, 10)

    return np.array([l1_next, l2_next])

# 2. Função de saída (o que queremos observar do sistema)
def double_tank_output(t, x, u, params):
    return np.array([x[0], x[1]])  # Retorna ambos os estados como saída



class CascadeWaterTankEnv(BaseSetPointEnv):
    """
    Environment specific for Cascade Water Tank.
    This class defines state and action spaces.


    #### Caso x_size != 2:

    Será necessário sobrescrever a função simulation_model com um wrapper.
    self.env.unwrapped.simulation_model = new_simulation_model
    """

    def __init__(
            self,

            # Base class parameters
            scheduller: Scheduller,
            ensemble_params: dict[str, np.float64],
            termination_rule: TerminationRule | Callable = TerminationRule.INTERVALS,
            error_formula: ErrorFormula | Callable = ErrorFormula.DIFFERENCE,
            reward_formula: RewardFormula | Callable = RewardFormula.DIFFERENCE_SQUARED,
            action_size: int = 1,
            x_size: int = 2,
            x_start_points: Optional[list[np.float64]] = None,
            tracked_point: str = 'x2',
            integrator_clip_bounds: Optional[tuple[float, float]] = (-25, 25),
            render_mode: Literal["terminal"] = "terminal",


            # Created parameters
            action_bounds: tuple[float, float] = (0, 1),            #[V]
            observation_max_bounds: tuple[float, float] = [10, 10], # [cm]
            ):
        
        # TODO criar wrapper que verifica se esse ambiente foi criado corretamente
        # assert action_size == 1 and x_size == 2, ""
        
        # ACTION_BOUNDARIES: list[tuple[float, float]] = [
        #     (-1_000, 1_000) # First action (u_t) 
        # ]

        assert x_start_points is None or x_size == len(x_start_points), \
            "Lenght of start_points must be equal to x_size."
        assert x_size == len(observation_max_bounds), \
            "Lenght of observation_max_boundaries must be equal to x_size."
        assert action_size == 1, \
            "action_size must be equal to 1."
        # assert action_size == len(ACTION_BOUNDARIES), \
        #     "Lenght of action_boundaries must be equal to action_size."
        
        super().__init__(
            scheduller=scheduller,
            start_ensemble_params=ensemble_params,
            termination_rule=termination_rule,
            error_formula=error_formula,
            reward_formula=reward_formula,
            action_size=1,
            x_size=x_size,
            x_start_points=x_start_points,
            tracked_point=tracked_point,
            integrator_clip_bounds=integrator_clip_bounds,
            render_mode=render_mode,
        )

        self.observation_max_bounds = observation_max_bounds
        
        # Definindo o espaço de ações (u_t)
        self.action_space = spaces.Box(
            low=action_bounds[0], 
            high=action_bounds[1], 
            shape=(action_size,), 
            dtype=np.float64
        )


        # x_vector = {}
        # for i in range(x_size):
        #     x_vector[f"x{i+1}"] = spaces.Box(low=0, high=observation_max_bounds[i], shape=(1,), dtype=np.float64)
        # x_vector[f"x{x_size+1}"] = self.action_space # Last action

        # Definindo o espaço de observações (l1_t e l2_t)
        self.observation_space = spaces.Dict({
            "x1": spaces.Box(low=0, high=observation_max_bounds[0], shape=(1,), dtype=np.float64),
            "x2": spaces.Box(low=0, high=observation_max_bounds[1], shape=(1,), dtype=np.float64),
            # "x3": self.action_space # Last action
            "y_ref": spaces.Box(low=0, high=observation_max_bounds[1], shape=(1,), dtype=np.float64),
            # "e_t": spaces.Box(low=float('-inf'), high=float('inf'), shape=(1,), dtype=np.float64),
            "z_t": spaces.Box(low=float('-inf'), high=float('inf'), shape=(1,), dtype=np.float64)
        })

        self.current_time = 0


    # uncomment if needed
    # def reset(self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None) -> tuple[dict[str, Any], dict]:
    #     return super().reset(seed=seed, options=options)

    def reset(self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        obs, _ = super().reset(seed=seed, options=options)

        # obs["x1"] = 0
        # obs["x2"] = 0
        # obs["x3"] = 0 # Consider last action as no volt applyed
        
        # 3. Criar o sistema não linear discreto usando `dt=2`
        self.double_tank_system = ct.nlsys(
            double_tank_update, double_tank_output, 
            states=2, inputs=1, outputs=2, 
            params=options["ensemble_sample_parameters"],#{"g": 981, "p1": 0.0020, "p2": 0.0020, "p3": 0.1, "dt": 2}, 
            dt=2  # Agora o sistema é discreto com passo 2 segundos
        )
        
        return obs, _


    def simulation_model(self,
                         u_t,                     # action
                         l1_t, l2_t,              # x vector
                        #  last_action,
                         /, *, 
                         g, p1, p2, p3, dt # Generated by ensemble
                        ) -> dict[str, np.float64]:
        
        """ 
        #### Caso x_size != 2: 

        então é necessário usar um  wrapper que sobrescreve simulation_model
        self.env.unwrapped.simulation_model = new_simulation_model
        
        #### Variáveis

        - l1_t, l2_t -> Nível da água;
        - a1, a2 -> Áreas dos buracos;
        - A1, A2 -> Áreas da seção transversal;
        - K_pump -> Constante da bomba d'água;
        - g -> Aceleração gravitacional;
        - p1 -> (a1 / A1) -> área dividido por área;
        - p2 -> (a2 / A2) -> área dividido por área;
        - p3 -> (K_pump / A1) -> constante dividido por área;
        - u_t -> Ação; Voltagem aplicada a bomba d'água; 
        - dt -> Intervalo de tempo; discretização de Euler do modelo;
        """

        t_out, y_out, x_out = ct.input_output_response(
            self.double_tank_system, 
            [self.current_time, self.current_time + dt], 
            np.array([u_t, 0]), 
            [l1_t, l2_t], 
            return_x=True
        )
        l1_t, l2_t = x_out[:, -1]

        self.current_time += dt

        # Considera bomba d'água não reversível, isto é, (u_t < 0) tem o mesmo efeito que (u_t = 0)
        # u_t = max(0, u_t)
        
        # delta_l1_t = (-p1 * np.sqrt(2 * g * l1_t) + p3 * u_t) * dt
        # l1_t = np.clip(l1_t + delta_l1_t, 0, 10) # 10 = self.observation_max_bounds[0]
        
        # delta_l2_t = (p1 * np.sqrt(2 * g * l1_t) - p2 * np.sqrt(2 * g * l2_t)) * dt
        # l2_t = np.clip(l2_t + delta_l2_t, 0, 10) # 10 = self.observation_max_bounds[1]
        
        # Lembrar: Modificar vetor x de simulation_model implica em modificar o x_size no create_water_tank_environment, o observation space no __init__ e o reset desse env
        return {
            "x1": l1_t, 
            "x2": l2_t,
            # "x3": u_t,      # last action
        }
    

    @staticmethod
    def create_water_tank_environment(seed = 42,
                                      set_points: list[float] = [3, 6, 9, 4, 2],
                                      intervals: list[float] = [400, 400, 400, 400, 400],
                                      distributions: dict[str, tuple[str, dict[str, float]]] = {
                                        "g": ("constant", {"constant": 981}),                # g (gravity) [cm/s²]
                                        "p1": ("uniform", {"low": 0.0015, "high": 0.0024}),  # p1
                                        "p2": ("uniform", {"low": 0.0015, "high": 0.0024}),  # p2
                                        "p3": ("uniform", {"low": 0.07, "high": 0.17}),      # p3
                                        "dt": ("constant", {"constant": 1}),                 # dt sample time [s]
                                      },
                                      integrator_bounds: tuple[float, float] = (-25, 25),
                                      agent_action_bounds: tuple[float, float] = (0, 1),
                                      agent_observation_min_bounds: tuple[float, float] = (0, 0),
                                      agent_observation_max_bounds: tuple[float, float] = (10, 10),
                                      pid_type: Literal["PID", "PI", "P"] = "PI",
                                      ) -> tuple[BaseSetPointEnv, Scheduller, EnsembleGenerator, Callable]:
        """ ## Variable Glossary

        s
            Variável complexa de Laplace. 
            Usada para representar a frequência em análises de sistemas no domínio de Laplace.

        g
            Unidade de medida: [cm/s²]
            Gravity. Gravidade.
        p1  
            Unidade de medida: adimensional
            Razão entre a área do abertura do tanque superior e a área da seção transversal do tanque superior.  
            p1 = a1 / A1, onde a1 é a área do abertura e A1 é a área do tanque.  

        p2  
            Unidade de medida: adimensional
            Razão entre a área do abertura do tanque inferior e a área da seção transversal do tanque inferior.  
            p2 = a2 / A2, onde a2 é a área do abertura e A2 é a área do tanque.

        p3  
            Unidade de medida: [m/s⋅V]
            Razão entre a constante da bomba e a área do tanque superior.  
            p3 = Kpump / A1, onde Kpump [m³/s⋅V] é a constante da bomba e A1 [m²] é a área do tanque superior.
        """
        
        scheduller = Scheduller(set_points, intervals) 
        ensemble = EnsembleGenerator(distributions, seed)

        env = gymnasium.make("CascadeWaterTankEnv-V0", 
                        scheduller             = scheduller,
                        ensemble_params        = ensemble.generate_sample(), 
                        termination_rule       = TerminationRule.INTERVALS,
                        error_formula          = ErrorFormula.DIFFERENCE,
                        reward_formula         = RewardFormula.DIFFERENCE_SQUARED,
                        action_size            = 1,
                        x_size                 = 2,
                        x_start_points         = None, #  [0, 0],
                        tracked_point          = 'x2',
                        integrator_clip_bounds = integrator_bounds,
                        action_bounds          = agent_action_bounds,
                        observation_max_bounds = agent_observation_max_bounds,
                        )
        env = DictToArrayWrapper(env)
        
        # # Define model symbols
        # t = sp.symbols('t', real=True, positive=True)
        # s = sp.symbols('s')
        # l1_t = sp.Function('l1_t')(t)
        # l2_t = sp.Function('l2_t')(t)
        # u_t = sp.Function('u_t')(t)
        # g, p1, p2, p3 = sp.symbols('g p1 p2 p3')
        # l1_s = sp.Function('L1')(s)
        # l2_s = sp.Function('L2')(s)
        # u_s = sp.Function('U')(s)

        # # Define model equations
        # delta_l1_t = -p1 * sp.sqrt(2 * g * l1_t) + p3 * u_t
        # delta_l2_t = p1 * sp.sqrt(2 * g * l1_t) - p2 * sp.sqrt(2 * g * l2_t)
        
        # # Apply Laplace transform
        # l1_s = sp.laplace_transform(l1_t, t, s, noconds=True)
        # l2_s = sp.laplace_transform(l2_t, t, s, noconds=True)
        # u_s = sp.laplace_transform(u_t, t, s, noconds=True)
        # delta_l1_t_laplace = sp.Eq(s * l1_s, -p1 * sp.sqrt(2 * g ) * l1_s + p3 * u_s)
        # delta_l2_t_laplace = sp.Eq(s * l2_s, p1 * sp.sqrt(2 * g) * l1_s - p2 * sp.sqrt(2 * g ) * l2_s)
        # l1_s_solution = sp.solve(delta_l1_t_laplace, l1_s)[0]
        # l1_s_solution = sp.simplify(l1_s_solution)
        # l2_s_solution = sp.solve(delta_l2_t_laplace.subs(l1_s, l1_s_solution), l2_s)[0] / u_s
        # water_tank_model = sp.simplify(l2_s_solution)

        # # Injeta parâmetros e converte para ct.transferFunction
        # parameters_values = ensemble.generate_sample()
        # water_tank_model = water_tank_model.subs(parameters_values)
        # num, den = sp.fraction(water_tank_model)
        # num = [float(coef.evalf()) for coef in sp.Poly(num, s).all_coeffs()]
        # den = [float(coef.evalf()) for coef in sp.Poly(den, s).all_coeffs()]
        # water_tank_model = ct.TransferFunction(num, den)

        # trained_pid, pid_optimized_params = PIDController.train_pid_controller(
        #     plant=water_tank_model, 
        #     pid_training_method='ZN',
        #     pid_type=pid_type,
        # )

        trained_pid = None
        pid_optimized_params = None

        return env, scheduller, ensemble, trained_pid, pid_optimized_params
