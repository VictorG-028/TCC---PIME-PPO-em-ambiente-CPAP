from datetime import datetime
import locale

from environments.CPAP_env import CpapEnv
from environments.cascade_water_tank_env import CascadeWaterTankEnv
from environments.ph_control_env import PhControl

from optuna_training import run_optuna
from training import run_training

################################################################################

# Set locale to pt_BR to show time in portuguese
locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
current_date_time = datetime.now().strftime("%d-%m-%H%M") # day-month-hourminute
base_dir = "logs/ppo/{env_name}/"+current_date_time

################################################################################

experiments = {
    'double_water_tank': {
        'create_env_function': CascadeWaterTankEnv.create_water_tank_environment,
        'logs_folder_path': base_dir.format(env_name="double_water_tank"),

        "hyperparameters": {
            "extra info": "função de simulação control", # "função de simulação usando biblioteca control",
            "seed": 42,
            "PIME_PPO": {
                ## PPO
                "vf_coef": 1.0, # Optuna nesse [0.01 até 10.0]
                "ent_coef": 0.02,
                "gae_lambda": 0.97,
                "clip_range": 0.2,
                "discount": 0.995,
                "horizon": 200, # Optuna nesse # pode variar até valor grande (1000)
                "adam_stepsize": 3e-4, # Optuna nesse [float entre e-2 até e-5]
                "minibatch_size": 256, # Optuna nesse # 256 valor grande # usar 2^n
                "epochs": 10,
                "ensemble_size": 1, # usar optuna com ensemble_size == 1
                "divide_neural_network": False,
                "neurons_per_layer": 100,
                "activation_function_name": "no activation",

                ## PIME
                "tracked_point_name": 'x2',
                "episodes_per_sample": 5, # esse parâmetro tanto faz pro optuna

                # PID (ZN - kp=8.80 e ki=11.46)
                "Kp": 8.8, 
                "Ki": 0.015,
                "Kd": 0,
            },
            "PID_and_Env": {
                "integrator_bounds": (-25, 25),
                "ppo_action_bounds": (-1, 1),           # [V]
                "ppo_observation_max_bounds": (10, 10), # [cm]
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
        'logs_folder_path': base_dir.format(env_name="ph_control"),
        'tracked_point': 'x2',
    },
    'CPAP': {
        'create_env_function': CpapEnv.create_cpap_environment,
        'logs_folder_path': base_dir.format(env_name="CPAP"),

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

                ## PIME
                "tracked_point_name": 'x3',
                "episodes_per_sample": 5,

                # PID
                "Kp": 2.5,
                "Ki": 0.030,
                "Kd": 0,
            },
            "PID_and_Env": {
                "integrator_bounds": (-25, 25),
                "ppo_action_bounds": (0, 100),        # [cmH2O]
                "pid_type": "PI",
            },
            ## Scheduller
            "set_points": [5, 15, 10],
            "intervals": [500, 500, 500],

            ## Model Ensemble # TODO encontrar distribuições diferentes de "constant" para esses parâmetros
            "distributions": {
                # Pacient (not used)
                "rp": ("constant", {"constant": 10e-3}),
                "c": ("constant", {"constant": 50}),
                "rl": ("constant", {"constant": 48.5 * 60 / 1000}),

                # Blower (not used)
                "tb": ("constant", {"constant":  10e-3}),
                "kb": ("constant", {"constant": 0.5}),

                # Lung
                "r_aw": ("constant", {"constant": 3}),
                # "_r_aw": ("constant", {"constant": 3 / 1000}),
                "c_rs": ("constant", {"constant": 60}),

                # Ventilator
                "v_t": ("constant", {"constant": 350}),
                "peep": ("constant", {"constant": 5}),
                "rr": ("constant", {"constant": 15}),
                "t_i": ("constant", {"constant": 1}),
                "t_ip": ("constant", {"constant": 0.25}),

                # Calculated inside Ventilator with above variables (not needed)
                # "_rr": ("constant", {"constant": 15 / 60}),
                # "t_c": ("constant", {"constant": 1 / (15 / 60)}),
                # "t_e": ("constant", {"constant": (1 / (15 / 60)) - 1 - 0.25}),
                # "_f_i": ("constant", {"constant": 350 / 1}),

                # Model constants
                "f_s": ("constant", {"constant": 30}),    # [Hz]
                "dt": ("constant", {"constant": 1/30}),   # [s]
            }
        },
    },
}

################################################################################

# run_training(experiments["double_water_tank"])
run_optuna(experiments["double_water_tank"])

