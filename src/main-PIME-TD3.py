import locale
import sys

import numpy as np

from algorithms.PIME_TD3 import PIME_TD3
from environments.CPAP_env import CpapEnv
from environments.cascade_water_tank_env import CascadeWaterTankEnv
from environments.ph_control_env import PhControl

from training import run_training
from optuna_training import run_optuna

################################################################################

locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8') # show date and time in portuguese format

################################################################################

experiments = {
    'double_water_tank': {
        "create_env_function": CascadeWaterTankEnv.create_water_tank_environment,
        "training_logs_folder_path": "logs/TD3_DDPG/double_water_tank/final/{id}",
        "best_results_folder_path": "logs/TD3_DDPG/double_water_tank/best/",

        "hyperparameters": {
            "extra info": "função de simulação control", # "função de simulação usando biblioteca control",
            "seed": 42,
            "PIME": {
                ## TD3_DDPG
                "target_decay": 0.97,
                "discount": 0.995,
                "mu_lr": 2.3305045634697055e-05,
                "q_lr": 2.3305045634697055e-05,
                "batch_size": 256,
                "epochs": 16,
                "policy_update_freq": 32,
                "noise": 0.01,
                "noise_clip": 0.05,
                "ensemble_size": 100,
                "divide_neural_network": True,
                "neurons_per_layer": 6,
                "activation_function_name": "tanh", # "no activation" "relu" "tanh"
                "use_activation_func_in_last_layer": False,

                ## PIME
                "tracked_point_name": 'x2',
                "episodes_per_sample": 5,
                "env_name": "double_water_tank",

                # PID (ZN - kp=8.80 e ki=11.46) (Optuna - kp=9.31 e ki=4.31) (manual - kp=9.0 e ki=0.015)
                "Kp": 9.31, 
                "Ki": 0.015,
                "Kd": 0.0,
            },
            "PID_and_Env": {
                "integrator_bounds": (-25, 25),
                "agent_action_bounds": (-30, 30),              # [V]
                "agent_observation_min_bounds": (0, 0),     # [cm, cm]
                "agent_observation_max_bounds": (10, 10),   # [cm, cm]
                "pid_type": "PI",
            },
            ## Scheduller
            "set_points": [3,6,9,4,2], # [cm]
            "intervals": [400, 400, 400, 400, 400], # [s]

            ## Model Ensemble
            "distributions": {
                "g": ("constant", {"constant": 981}),                # g (gravity) [cm/s²]
                "p1": ("uniform", {"low": 0.0015, "high": 0.0024}),  # p1
                "p2": ("uniform", {"low": 0.0015, "high": 0.0024}),  # p2
                "p3": ("uniform", {"low": 0.07, "high": 0.17}),      # p3
                "dt": ("constant", {"constant": 2}),                 # dt sample time [s] # Antees tava 2 segundso
            },
        }
    },
    'CPAP': {
        "create_env_function": CpapEnv.create_cpap_environment,
        "training_logs_folder_path": "logs/TD3_DDPG/CPAP/final/{id}",
        "best_results_folder_path": "logs/TD3_DDPG/CPAP/best/",

        "hyperparameters": {
            "seed": 42,
            "PIME": {
                ## TD3_DDPG
                "target_decay": 0.97,
                "discount": 0.94,
                "mu_lr": 3.357890528561397e-05,
                "q_lr": 3.357890528561397e-05,
                "batch_size": 64,
                "epochs": 4,
                "policy_update_freq": 6,
                "noise": 0.2,
                "noise_clip": 0.5,
                "ensemble_size": 5,
                "divide_neural_network": True,
                "neurons_per_layer": 48,
                "activation_function_name": "tanh", # "no activation" "relu" "tanh"
                "use_activation_func_in_last_layer": False,

                ## PIME
                "tracked_point_name": 'x3',
                "episodes_per_sample": 5,
                "env_name": "CPAP",

                # PID
                "Kp": 10.0,
                "Ki": 1.2,
                "Kd": 0.0,
            },
            "PID_and_Env": {
                "integrator_bounds": (-25, 25),
                "agent_action_bounds": (-30, 30),                        # [v]
                "agent_observation_min_bounds": (-np.inf, -np.inf, -14), # [ml/s, ml^3, cmH2O]
                "agent_observation_max_bounds": (np.inf, np.inf, 14),    # [ml/s, ml^3, cmH2O]
                "pid_type": "PI",
                "max_step": 10_000,
            },
            ## Scheduller
            "set_points": [3, 12, 5, 7, 5], # pressure [cmH2O]
            "intervals": [2000, 2000, 2000, 2000, 2000], # [1/10 s]

            "distributions": {

                # Lung (parâmetros pulmonares)
                "r_aw": ("uniform", {"low": 2, "high": 5}),        # [cmH2O / L / s] - resistência normal: 2 a 5
                "c_rs": ("uniform", {"low": 35, "high": 60}),      # [ml / cmH2O] - complacência normal: 85 a 100

                # Ventilator (parâmetros do ventilador)
                "kv": ("constant", {"constant": 0.99}),            # [cm³/s/V]
                "tv": ("constant", {"constant": 0.01}),            # [s]
                
                # Pacient
                "v_t": ("uniform", {"low": 300, "high": 400}),     # [ml] - volume corrente: 300 a 400 ml
                "rr": ("uniform", {"low": 10, "high": 20}),        # [min^-1] - frequência respiratória: 10 a 20 respirações/min
                "t_i": ("uniform", {"low": 0.8, "high": 1.2}),     # [s] - tempo inspiratório: 0.8 a 1.2 s
                "t_ip": ("uniform", {"low": 0.1, "high": 0.3}),    # [s] - pausa inspiratória: 0.2 a 0.3 s

                # Filtro de passa baixa
                "tau_f": ("constant", {"constant": 0.1}),         # [s] - constante de tempo do filtro de passa baixa
                
                # Model constants
                "f_s": ("constant", {"constant": 1000}),           # [Hz] - frequência de amostragem
                "dt": ("constant", {"constant": 1/1000}),          # [s] 1/1000
            }
        },
    },
}

################################################################################


with open("./terminal_outputs.txt", 'w') as f:

    # sys.stdout = f

    run_training(
        PIME_TD3,
        experiments["double_water_tank"], 
        steps_to_run=400_000,
        should_save_records=True,
        extra_record_only_pid=True,
        should_save_trained_model=True,
        extra_record_trained_agent=True,
        use_GPU=False
    )

    # run_training(
    #     PIME_TD3,
    #     experiments["CPAP"], 
    #     steps_to_run=400_000,
    #     should_save_records=True,
    #     extra_record_only_pid=False,
    #     should_save_trained_model=True,
    #     extra_record_trained_agent=False,
    #     use_GPU=False
    # )

    # run_optuna(
    #     experiments["CPAP"], 
    #     steps_to_run=200_000, # 2_000, # 15_000,
    #     should_save_records=True,
    #     extra_record_only_pid=False,
    #     should_save_trained_model=False,
    #     n_trials=17*10,
    #     n_jobs=17,
    #     use_GPU=False
    # )

    sys.stdout = sys.__stdout__



