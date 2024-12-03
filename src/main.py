from typing import Callable
from algorithms.PIME_PPO import PIME_PPO
# from CustomEnv.WaterTankEnv import WaterTankENv
# from CustomEnv.PHEnv import PHENv
from wrappers.DictToArray import DictToArrayWrapper
from enums.TerminationRule import TerminationRule
from enums.ErrorFormula import ErrorFormula

from modules.Ensemble import Ensemble
from modules.Scheduller import Scheduller

from environments import BaseSetPointEnv

import gymnasium
import numpy as np
from stable_baselines3 import PPO

from simulation_models.cascaded_water_tanks import simulation_model as double_tank_simultaion
from simulation_models.ph_control import simulation_model as ph_simulation
from simulation_models.CPAP import simulation_model as cpap_simulation

import sympy as sp
import control as ct
from scipy.optimize import minimize

################################################################################

def train_pid_controller(
                                plant: ct.TransferFunction, 
                                pid_training_method: str = 'BFGS', 
                                initial_kp = 0, 
                                initial_ki = 0, 
                                initial_kd = 0
                            ) -> Callable[[float], float]:
        """
        Train a PID controller for a given plant using a specified optimization method.

        Parameters:
        plant (Callable[[np.ndarray], np.ndarray]): The plant to be controlled.
        pid_training_method (str): The optimization method to use for training the PID controller.

        Returns:
        function: A function that takes an input array and returns the control signal from the trained PID controller.
        """

        def optimize_pid(params):
            Kp, Ki, Kd = params
            pid = ct.TransferFunction([Kd, Kp, Ki], [1, 0.001, 0.001])
            closed_loop = ct.feedback(pid * plant, 1)
            t, y = ct.step_response(closed_loop, T=np.linspace(0, 10, 1000))
            # Objective: minimize the integral of the absolute error
            error = 1 - y  # assuming unit step input
            return np.sum(np.abs(error))

        
        # Perform pid optimization
        optimized_params = minimize(
            optimize_pid, 
            [initial_kp, initial_ki, initial_kd], 
            method = pid_training_method
        ) 

        optimized_Kp, optimized_Ki, optimized_Kd = optimized_params.x
        
        optimized_pid = ct.TransferFunction(
            [optimized_Kd, optimized_Kp, optimized_Ki], 
            [1, 0.001, 0.001]
        )
        # print(f"Numerador: {optimized_pid.num}")
        # print(f"Denominador: {optimized_pid.den}")
        # print(f"{optimized_pid=}")
        # print(f"{optimized_pid.damp()=}")

        def pid_controller(error: float) -> float:
            """
            Compute the control signal using the trained PID controller.
            """
            pid_action = ct.forced_response(
                optimized_pid, 
                # T = np.array([0, 1e-6]), 
                T = np.array([0, 1]), # TODO revisar essa implementação
                U = np.array([0, error])
            )

            return pid_action.y[0, 1]  # Retornar o último valor da resposta

        return pid_controller, {'optimized_Kp': optimized_Kp, 'optimized_Ki': optimized_Ki, 'optimized_Kd': optimized_Kd}

################################################################################

def create_water_tank_environment() -> tuple[BaseSetPointEnv, Scheduller, Ensemble, Callable]:

    """ Variable Glossary

    s
        Variável complexa de Laplace. 
        Usada para representar a frequência em análises de sistemas no domínio de Laplace.

    # 
    g
        Unidade de medida: [cm/s²] # TODO confirmar unidade de medida
        Gravity. Gravidade.
    p1
        TODO
    p2
        TODO
    p3
        TODO
    """
     
    set_points = [5, 15, 10]
    intervals = [5, 5, 5]
    scheduller = Scheduller(set_points, intervals) 
    distributions = {
        "g": ("constant", {"constant": 981}),                # g (gravity)
        "p1": ("uniform", {"low": 0.0015, "high": 0.0024}),  # p1
        "p2": ("uniform", {"low": 0.0015, "high": 0.0024}),  # p2
        "p3": ("uniform", {"low": 0.07, "high": 0.17})       # p3
    }
    seed = 42
    ammount_of_parameter_sets = 2
    ensemble = Ensemble(ammount_of_parameter_sets, distributions, seed)

    env = gymnasium.make("CascadeWaterTankEnv-V0", 
                    scheduller             = scheduller, 
                    simulation_model       = double_tank_simultaion,
                    ensemble_params        = ensemble.get_param_set(), 
                    action_size            = 1,
                    x_size                 = 2,
                    x_start_points         = None,#  [20, 20],
                    tracked_point          = 'x2',
                    termination_rule       = TerminationRule.INTERVALS,
                    error_formula          = ErrorFormula.DIFFERENCE,
                    )
    env = DictToArrayWrapper(env)
    
    scheduller = Scheduller(set_points, intervals) 
    
    # Define model symbols
    t = sp.symbols('t', real=True, positive=True)
    s = sp.symbols('s')
    l1_t = sp.Function('l1_t')(t)
    l2_t = sp.Function('l2_t')(t)
    u_t = sp.Function('u_t')(t)
    g, p1, p2, p3 = sp.symbols('g p1 p2 p3')
    l1_s = sp.Function('L1')(s)
    l2_s = sp.Function('L2')(s)
    u_s = sp.Function('U')(s)

    # Define model equations
    delta_l1_t = -p1 * sp.sqrt(2 * g * l1_t) + p3 * u_t
    delta_l2_t = p1 * sp.sqrt(2 * g * l1_t) - p2 * sp.sqrt(2 * g * l2_t)
    
    # Apply Laplace transform
    l1_s = sp.laplace_transform(l1_t, t, s, noconds=True)
    l2_s = sp.laplace_transform(l2_t, t, s, noconds=True)
    u_s = sp.laplace_transform(u_t, t, s, noconds=True)
    delta_l1_t_laplace = sp.Eq(s * l1_s, -p1 * sp.sqrt(2 * g) * l1_s + p3 * u_s)
    delta_l2_t_laplace = sp.Eq(s * l2_s, p1 * sp.sqrt(2 * g) * l1_s - p2 * sp.sqrt(2 * g) * l2_s)
    l1_s_solution = sp.solve(delta_l1_t_laplace, l1_s)[0]
    l1_s_solution = sp.simplify(l1_s_solution)
    l2_s_solution = sp.solve(delta_l2_t_laplace.subs(l1_s, l1_s_solution), l2_s)[0] / u_s
    water_tank_model = sp.simplify(l2_s_solution)

    # Injeta parâmetros e converte para ct.transferFunction
    parameters_values = ensemble.get_all_samples()[0]
    water_tank_model = water_tank_model.subs(parameters_values)
    num, den = sp.fraction(water_tank_model)
    num = [float(coef.evalf()) for coef in sp.Poly(num, s).all_coeffs()]
    den = [float(coef.evalf()) for coef in sp.Poly(den, s).all_coeffs()]
    water_tank_model = ct.TransferFunction(num, den)

    trained_pid, pid_optimized_params = train_pid_controller(
        plant=water_tank_model, 
        pid_training_method='BFGS', 
        initial_kp=0, 
        initial_ki=0, 
        initial_kd=0
    )
    
    return env, scheduller, ensemble, trained_pid, pid_optimized_params


def create_ph_control_environment() -> tuple[BaseSetPointEnv, Scheduller, Ensemble, Callable]:

    """ Variable Glossary

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
    }
    seed = 42
    ammount_of_parameter_sets = 10
    ensemble = Ensemble(ammount_of_parameter_sets, distributions, seed)

    env = gymnasium.make("PhResidualWaterTreatmentEnv-V0", 
                    scheduller             = scheduller, 
                    simulation_model       = ph_simulation,
                    ensemble_params        = ensemble.get_param_set(), 
                    action_size            = 1,
                    x_size                 = 2,
                    x_start_points         = None,
                    tracked_point          = 'x2',
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
    parameters_values = ensemble.get_all_samples()[0]
    ph_control_model.subs(parameters_values)
    num, den = sp.fraction(water_tank_model)
    num = [float(coef.evalf()) for coef in sp.Poly(num, s).all_coeffs()]
    den = [float(coef.evalf()) for coef in sp.Poly(den, s).all_coeffs()]
    water_tank_model = ct.TransferFunction(num, den)

    trained_pid, pid_optimized_params = train_pid_controller(
        plant=ph_control_model, 
        pid_training_method='BFGS', 
        initial_kp=0, 
        initial_ki=0, 
        initial_kd=0
    )

    return env, scheduller, ensemble, trained_pid, pid_optimized_params


def create_cpap_environment() -> tuple[BaseSetPointEnv, Scheduller, Ensemble, Callable]:

    """ Variable Glossary

    s
        Variável complexa de Laplace. 
        Usada para representar a frequência em análises de sistemas no domínio de Laplace.

    # Pacient variables
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

    # Blower variables
    tb
        Unidade de medida: [s]
        Blower constant time. Constante de tempo do soprador.
        Representa o tempo necessário para o soprador atingir uma fração significativa de sua resposta final.
    kb
        Unidade de medida: [cmH2O/(L/s)] # TODO: Confirmar essa unidade de medida
        Blower Gain. Ganho do sporador.

    # PID variables
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
    _rp, _c, _rl = patient['Heated Humidifier, Normal'].values()
    _tb = 10e-3
    _kb = 0.5

    set_points = [5, 15, 10]
    intervals = [500, 500, 500]
    scheduller = Scheduller(set_points, intervals)

    # TODO encontrar distribuições diferentes de "constant" para esses parâmetros
    distributions = {
        # Pacient
        "rp": ("constant", {"constant": _rp}),
        "c": ("constant", {"constant": _c}),
        "rl": ("constant", {"constant": _rl}),

        # Blower
        "tb": ("constant", {"constant": _rp}),
        "kb": ("constant", {"constant": _kb}),

        # Lung
        "r_aw": ("constant", {"constant": 3}),
        # "_r_aw": ("constant", {"constant": 3 / 1000}),
        "c_rs": ("constant", {"constant": 60}),

        # Ventilator
        "v_t": ("constant", {"constant": 350}),
        "peep": ("constant", {"constant": 5}),
        "rr": ("constant", {"constant": 15}),
        # "_rr": ("constant", {"constant": 15 / 60}),
        "t_i": ("constant", {"constant": 1}),
        "t_ip": ("constant", {"constant": 0.25}),
        # "t_c": ("constant", {"constant": 1 / (15 / 60)}),
        # "t_e": ("constant", {"constant": (1 / (15 / 60)) - 1 - 0.25}),
        # "_f_i": ("constant", {"constant": 350 / 1}),
    }
    seed = 42
    ammount_of_parameter_sets = 2
    ensemble = Ensemble(ammount_of_parameter_sets, distributions, seed)
    
    env = gymnasium.make("CpapEnv-V0", 
                    scheduller             = scheduller, 
                    simulation_model       = cpap_simulation,
                    ensemble_params        = ensemble.get_param_set(), 
                    x_size                 = 3,
                    x_start_points         = None,
                    tracked_point          = 'x3',
                    termination_rule       = TerminationRule.MAX_STEPS,
                    error_formula          = ErrorFormula.DIFFERENCE,
                    )
    env = DictToArrayWrapper(env)

    # Define model symbols
    s = sp.symbols('s')
    tb, kb = sp.symbols('tb kb')
    rp, rl, c = sp.symbols('rp rl c')
    # kp, ki, kd = sp.symbols('kp ki kd') # Not used

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
    trained_pid, pid_optimized_params = train_pid_controller(cpap_model, pid_training_method='BFGS')

    return env, scheduller, ensemble, trained_pid, pid_optimized_params

################################################################################

experiments = {
    'double_water_tank': {
        'create_env_function': create_water_tank_environment,
        'logs_folder_path': "logs/ppo/double_water_tank",
        'tracked_point': 'x2',
    },
    'ph_control': {
        'create_env_function': create_ph_control_environment,
        'logs_folder_path': "logs/ppo/ph_control",
        'tracked_point': 'x2',
    },
    'CPAP': {
        'create_env_function': create_cpap_environment,
        'logs_folder_path': "logs/ppo/CPAP",
        'tracked_point': 'x3',
    },
}

create_env_function, logs_folder_path, tracked_point = experiments['CPAP'].values()
env, scheduller, ensemble, trained_pid, pid_optimized_params = create_env_function()

# y = trained_pid(44.358696)
# print(f"{y=}")
# exit()
pime_ppo_controller = PIME_PPO(
                            env, 
                            scheduller, 
                            ensemble, 
                            # trained_pid,
                            **pid_optimized_params,
                            tracked_point_name=tracked_point,
                            logs_folder_path=logs_folder_path,
                            )

pime_ppo_controller.train(iterations = 10)
