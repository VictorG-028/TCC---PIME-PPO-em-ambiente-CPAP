import locale
import sys

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
        "training_logs_folder_path": "logs/ppo/double_water_tank/trainings/{id}",
        "best_results_folder_path": "logs/ppo/double_water_tank/best/",

        "hyperparameters": {
            "extra info": "função de simulação control", # "função de simulação usando biblioteca control",
            "seed": 42,
            "PIME_PPO": {
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
                "ppo_action_bounds": (-1, 1),           # [V]
                "ppo_observation_max_bounds": (10, 10), # [cm]
                "pid_type": "P",
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
        "training_logs_folder_path": "logs/ppo/CPAP/trainings/{id}",
        "best_results_folder_path": "logs/ppo/CPAP/best/",

        "hyperparameters": {
            "seed": 42,
            "PIME_PPO": {
                ## PPO
                "vf_coef": 1.0,
                "ent_coef": 0.02,
                "gae_lambda": 0.97,
                "clip_range": 0.2,
                "discount": 0.99,
                "horizon": 200,
                "adam_stepsize": 3e-4,
                "minibatch_size": 256,
                "epochs": 10,
                "ensemble_size": 2,
                "divide_neural_network": True,
                "neurons_per_layer": 6,
                "activation_function_name": "no activation", # "no activation" "relu" "tanh"

                ## PIME
                "tracked_point_name": 'x3',
                "episodes_per_sample": 5,

                # PID
                "Kp": 3.0,
                "Ki": 0.0030,
                "Kd": 0,
            },
            "PID_and_Env": {
                "integrator_bounds": (-25, 25),
                "ppo_action_bounds": (-1, 1),           # [v]
                "ppo_observation_min_bounds": (10, 10), # [?]
                "ppo_observation_max_bounds": (10, 10), # [?]
                "pid_type": "PI",
            },
            ## Scheduller
            "set_points": [3, 12, 5, 7, 5], # pressure [cmH2O]
            "intervals": [50, 50, 50, 50, 50], # [s]

            "distributions": {
                # Blower (saída de fluxo volumétrico [cm³/s])
                # "tb": ("uniform", {"low": 8e-3, "high": 1.2e-2}),  # [s]
                # "kb": ("uniform", {"low": 0.4, "high": 0.6}),      # [cm³/s/V]

                # # Lung (parâmetros pulmonares)
                # "r_aw": ("uniform", {"low": 2, "high": 5}),        # [cmH2O / L / s] - resistência normal: 2 a 5
                # "c_rs": ("uniform", {"low": 85, "high": 100}),     # [ml / cmH2O] - complacência normal: 85 a 100

                # # Ventilator (parâmetros do ventilador)
                # "v_t": ("uniform", {"low": 300, "high": 400}),     # [ml] - volume corrente: 300 a 400 ml
                # "peep": ("uniform", {"low": 4, "high": 6}),        # [cmH2O] - pressão expiratória: 4 a 6 cmH2O
                # "rr": ("uniform", {"low": 10, "high": 20}),        # [min^-1] - frequência respiratória: 10 a 20 respirações/min
                # "t_i": ("uniform", {"low": 0.8, "high": 1.2}),     # [s] - tempo inspiratório: 0.8 a 1.2 s
                # "t_ip": ("uniform", {"low": 0.2, "high": 0.3}),    # [s] - pausa inspiratória: 0.2 a 0.3 s

                # Blower
                "tb": ("constant", {"constant": 7.0}),
                "kb": ("constant", {"constant": 0.05}),

                # Lung
                "r_aw": ("constant", {"constant": 3}),
                "c_rs": ("constant", {"constant": 60}),

                # Ventilator
                "v_t": ("constant", {"constant": 350}),
                # "peep": ("constant", {"constant": 1}),
                "rr": ("constant", {"constant": 15}),
                "t_i": ("constant", {"constant": 1}),
                "t_ip": ("constant", {"constant": 0.25}),

                # Model constants
                "f_s": ("constant", {"constant": 30}),             # [Hz] - frequência de amostragem
                "dt": ("constant", {"constant": 1/30}),            # [s] 1/30
            }
        },
    },
}

################################################################################


with open("./terminal_outputs.txt", 'w') as f:

    sys.stdout = f

    run_training(
        experiments["CPAP"], 
        steps_to_run=250, 
        should_save_records=True,
        extra_record_only_pid=True,
        should_save_trained_model=False,
        use_GPU=False
    )

    # run_optuna(
    #     experiments["double_water_tank"], 
    #     steps_to_run=400_000, 
    #     should_save_records=True,
    #     extra_record_only_pid=False,
    #     should_save_trained_model=False,
    #     n_trials=17*7, 
    #     n_jobs=17, 
    #     use_GPU=False
    # )

    sys.stdout = sys.__stdout__



