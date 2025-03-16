import locale
import sys

import numpy as np

from algorithms.PIME_TD3 import PIME_PPO
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
        "training_logs_folder_path": "logs/ppo/double_water_tank/final_test/{id}",
        "best_results_folder_path": "logs/ppo/double_water_tank/best/",

        "hyperparameters": {
            "extra info": "função de simulação control", # "função de simulação usando biblioteca control",
            "seed": 42,
            "PIME": {
                ## PPO
                "vf_coef":  1.0, 
                "ent_coef": 0.02,
                "gae_lambda": 0.97,
                "clip_range": 0.2,
                "discount": 0.995,
                "horizon": 200,
                "adam_stepsize": 3e-4,
                "minibatch_size": 128,
                "epochs": 10,
                "ensemble_size": 5,
                "divide_neural_network": True,
                "neurons_per_layer": 6,
                "activation_function_name": "no activation", # "no activation" "relu" "tanh"

                ## PIME
                "tracked_point_name": 'x2',
                "episodes_per_sample": 5,

                # PID (ZN - kp=8.80 e ki=11.46)
                "Kp": 8.8, 
                "Ki": 0.015,
                "Kd": 0,
            },
            "PID_and_Env": {
                "integrator_bounds": (-25, 25),
                "agent_action_bounds": (-1, 1),           # [V]
                "agent_observation_min_bounds": (0, 0), # [cm, cm]
                "agent_observation_max_bounds": (10, 10), # [cm, cm]
                "pid_type": "PI",
            },
            ## Scheduller
            "set_points": [3, 6, 9, 4, 2], # [cm]
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
    'ph_control': {
        'create_env_function': PhControl.create_ph_control_environment,
        'logs_folder_path': "logs/ppo/ph_control/{start_date_time}",
        'tracked_point': 'x2',
    },
    'CPAP': {
        "create_env_function": CpapEnv.create_cpap_environment,
        "training_logs_folder_path": "logs/ppo/CPAP/optuna_trainings_final/{id}",
        "best_results_folder_path": "logs/ppo/CPAP/best/",

        "hyperparameters": {
            "seed": 42,
            "PIME": {
                ## PPO
                "vf_coef": 1.0,
                "ent_coef": 0.02,
                "gae_lambda": 0.97,
                "clip_range": 0.2,
                "discount": 0.94,
                "horizon": 200,
                "adam_stepsize": 3e-4,
                "minibatch_size": 256,
                "epochs": 10,
                "ensemble_size": 5,
                "divide_neural_network": True,
                "neurons_per_layer": 6,
                "activation_function_name": "no activation", # "no activation" "relu" "tanh"

                ## PIME
                "tracked_point_name": 'x3',
                "episodes_per_sample": 5,

                # PID
                "Kp": 5.0,
                "Ki": 9.2,
                "Kd": 0,
            },
            "PID_and_Env": {
                "integrator_bounds": (-25, 25),
                "agent_action_bounds": (-1, 1),                          # [v]
                "agent_observation_min_bounds": (-np.inf, -np.inf, -14), # [ml/s, ml^3, cmH2O]
                "agent_observation_max_bounds": (np.inf, np.inf, 14),    # [ml/s, ml^3, cmH2O]
                "pid_type": "PI",
                "max_step": 10_000,
                # "max_step": 15_000,
            },
            ## Scheduller
            "set_points": [3, 12, 5, 7, 5], # pressure [cmH2O]
            "intervals": [2000, 2000, 2000, 2000, 2000], # [s]
            # "set_points": [7, 9, 5], # pressure [cmH2O]
            # "intervals": [5000, 5000, 5000], # [s]

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

                # # Lung
                # "r_aw": ("constant", {"constant": 3}),
                # "c_rs": ("constant", {"constant": 60}),

                # # Ventilator
                # "kv": ("constant", {"constant": 0.99}),
                # "tv": ("constant", {"constant": 0.01}),
                # "v_t": ("constant", {"constant": 350}),
                # "rr": ("constant", {"constant": 15}),
                # "t_i": ("constant", {"constant": 1.0}),
                # "t_ip": ("constant", {"constant": 0.25}),
            }
        },
    },
}

################################################################################


with open("./terminal_outputs.txt", 'w') as f:

    # sys.stdout = f
    import time
    start_time = time.time()

    run_training(
        PIME_PPO,
        experiments["double_water_tank"], 
        steps_to_run=1_000_000, 
        should_save_records=True,
        extra_record_only_pid=False,
        should_save_trained_model=False,
        use_GPU=False
    )
    # run_optuna(
    #     PIME_PPO,
    #     experiments["CPAP"], 
    #     steps_to_run=400_000, # 2_000, # 15_000,
    #     should_save_records=True,
    #     extra_record_only_pid=False,
    #     should_save_trained_model=False,
    #     n_trials=17*1,
    #     n_jobs=17,
    #     use_GPU=False
    # )
    end_time = time.time()
    print(f"Tempo de execução: {(end_time - start_time)=} segundos")
    # run_optuna(
    #     experiments["CPAP"], 
    #     steps_to_run=400_000, # 2_000, # 15_000,
    #     should_save_records=True,
    #     extra_record_only_pid=False,
    #     should_save_trained_model=False,
    #     n_trials=17*10,
    #     n_jobs=17,
    #     use_GPU=False
    # )

    sys.stdout = sys.__stdout__



