from algorithms.PID_Controller import PIDController
from environments.base_set_point_env import BaseSetPointEnv
from typing import Callable, Literal, Optional
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




class PhControl(BaseSetPointEnv):
    """
    Environment specific for residual water treatment by controling ph.
    This class defines state and action spaces.
    """

    def __init__(
            self,

            # Base class parameters
            scheduller: Scheduller,
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
            start_ensemble_params=ensemble_params,
            termination_rule=termination_rule,
            error_formula=error_formula,
            action_size=1,
            x_size=2,
            x_start_points=x_start_points,
            tracked_point="x1",
            render_mode=render_mode,
        )

        # Definindo o espaço de ações (u_t)
        self.action_space = spaces.Box(
            low=-1_000,
            high=1_000, 
            shape=(1,),
            dtype=np.float64
        )

        # Definindo o espaço de observações (ph_t e conc_t)
        self.observation_space = spaces.Dict({
            "x1": spaces.Box(low=0, high=14, shape=(1,), dtype=np.float64), # ph
            "x2": spaces.Box(low=0, high=1000, shape=(1,), dtype=np.float64), # conc
            "y_ref": spaces.Box(low=0, high=14, shape=(1,), dtype=np.float64),
            "z_t": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float64)
        })


    def simulation_model(u_t,                        # action
                         ph_t, hcl_concentration_t,  # x vector
                         /, *, 
                         p1, p2, dt: float
                        ) -> dict[str, np.float64]:
        """ 
        Modelo de simulação para controle de pH em uma estação de 
        tratamento de águas residuais.
        
        #### Glossário de variáveis

        - ph_t -> pH;
        - waste_conc_t -> Concentração de resíduos indesejados;
        - p1 -> ([HCL,C] * q_[HCL,C]) / V -> concentração de ácido clorídrico[HCl,C] vezes constante de "flow" de [HCl,C] dividido por volume do tanque de reação;
        - p2 -> q_ww / V -> taxa de fluxo de entrada de água residual q_ww dividido pelo volume do tanque de reação V;
        - K_reagent -> Constante de reagente, descrevendo a eficiência do reagente na neutralização do pH;
        - u_t -> Ação; Dosagem de reagente;
        - dt -> Intervalo de tempo; discretização de Euler do modelo;
        """

        # Atualiza a concentração de HCl usando o método de Euler
        delta_hcl_t = -p2 * hcl_concentration_t + p1 * u_t
        delta_hcl_t = delta_hcl_t # * dt

        # Evita valores negativos
        hcl_concentration_t = max(0, hcl_concentration_t + delta_hcl_t) 
        
        # Calcula o pH baseado na nova concentração de H+
        h_concentration_t = hcl_concentration_t  # Simplificação, assumindo [HCl] ≈ [H+]
        
        # Evitar log de zero, assumindo pH máximo de 14
        if h_concentration_t > 0:
            ph_t = -np.log10(h_concentration_t) 
        else:
            ph_t = 14
        
        return {"x1": ph_t, "x2": hcl_concentration_t}


    @staticmethod
    def create_ph_control_environment() -> tuple[BaseSetPointEnv, Scheduller, EnsembleGenerator, Callable]:
        """ ## Variable Glossary

        s
            Variável complexa de Laplace. 
            Usada para representar a frequência em análises de sistemas no domínio de Laplace.

        u_t
            Ação; Dosagem do reagente (aumenta a concentração da molécula HCL);
        ph_t
            É calculado usando a formula do paper x(t) = -log_10( [H+] )(t)
            Representa o valor do pH no timestep t.
        hcl_concentration_t
            TODO
        delta_hcl_t
            O quanto a concentração de HCL muda depois ao executar uma ação u_t.
        h_concentration_t
            [H+] = Concentração do ion H+ no timestep t.
        p1
            p1 = ([HCL,C] * q_[HCL,C]) / V
            [HCl,C] = concentração de ácido clorídrico 
            q_[HCL,C] = constante de "flow" de [HCl,C] 
            V = volume do tanque de reação
        p2
            p2 = q_ww / V 
            q_ww = taxa de fluxo de entrada de água residual
            V = volume do tanque de reação
        k_reagent
            [!] não usado na versão atual
            Representa a eficiência do reagente na neutralização do pH
        """

        set_points = [5, 15, 10]
        intervals = [5, 5, 5]
        scheduller = Scheduller(set_points, intervals) 
        distributions = {
            "p1": ("uniform", {"low": 0.005, "high": 0.015}),
            "p2": ("uniform", {"low": 0.0015, "high": 0.0025}),
            "dt": ("constant", {"constant": 20}),                 # dt sample time (seconds)
        }
        seed = 42
        ensemble = EnsembleGenerator(distributions, seed)

        env = gymnasium.make("PhControlEnv-V0", 
                        scheduller             = scheduller,
                        ensemble_params        = ensemble.generate_sample(), 
                        # action_size            = 1,
                        # x_size                 = 2,
                        x_start_points         = None,
                        # tracked_point          = 'x2',
                        termination_rule       = TerminationRule.INTERVALS,
                        error_formula          = ErrorFormula.DIFFERENCE,
                        )
        env = DictToArrayWrapper(env)
        
        scheduller = Scheduller(set_points, intervals) 

        # Define model symbols
        t = sp.symbols('t', real=True, positive=True)
        s = sp.symbols('s')
        hcl_concentration_t = sp.Function('hcl_concentration_t')(t)
        u_t = sp.Function('u_t')(t)
        p1, p2 = sp.symbols('p1 p2')
        hcl_s = sp.Function('HCL')(s)
        u_s = sp.Function('U')(s)
        delta_hcl_t = -p2 * hcl_concentration_t + p1 * u_t


        # Calculate the model transfer function using laplace transform
        hcl_s = sp.laplace_transform(hcl_concentration_t, t, s, noconds=True)
        u_s = sp.laplace_transform(u_t, t, s, noconds=True)
        delta_hcl_t_laplace_transformed = sp.Eq(s * hcl_s, -p2 * hcl_s + p1 * u_s)
        ph_control_model = sp.solve(delta_hcl_t_laplace_transformed, hcl_s)[0] / u_s
        ph_control_model = sp.simplify(ph_control_model)
        
        # Injeta parâmetros e converte para ct.transferFunction
        parameters_values = ensemble.generate_sample()
        ph_control_model.subs(parameters_values)
        # num, den = sp.fraction(ph_control_model)
        # num = [float(coef.evalf()) for coef in sp.Poly(num, s).all_coeffs()]
        # den = [float(coef.evalf()) for coef in sp.Poly(den, s).all_coeffs()]
        # ph_control_model = ct.TransferFunction(num, den)

        # trained_pid, pid_optimized_params = PIDController.train_pid_controller(
        #     plant=ph_control_model, 
        #     pid_training_method='BFGS', 
        #     initial_kp=0, 
        #     initial_ki=0, 
        #     initial_kd=0
        # )

        return env, scheduller, ensemble, None, {"optimized_Kp": 1, "optimized_Ki": 1, "optimized_Kd": 0}
        # return env, scheduller, ensemble, trained_pid, pid_optimized_params
