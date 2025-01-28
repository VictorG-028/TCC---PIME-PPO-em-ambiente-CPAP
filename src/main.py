from datetime import datetime
import locale

from algorithms.PIME_PPO import PIME_PPO
from environments.CPAP_env import CpapEnv
from environments.cascade_water_tank_env import CascadeWaterTankEnv
from environments.ph_control_env import PhControl
from modules.SaveFiles import save_hyperparameters_as_json

locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
current_date_time = datetime.now().strftime("%d-%m-%H%M") # day-month-hourminute
base_dir = "logs/ppo/{env_name}/"+current_date_time

experiments = {
    'double_water_tank': {
        'create_env_function': CascadeWaterTankEnv.create_water_tank_environment,
        'logs_folder_path': base_dir.format(env_name="double_water_tank"),

        "hyperparameters": {
            "seed": 42,
            "PIME_PPO": {
                ## PPO
                "vf_coef": 0.02,
                "ent_coef": 1.0,
                "gae_lambda": 0.97,
                "clip_range": 0.2,
                "gamma": 0.995,
                "horizon": 200,
                "adam_stepsize": 3e-4,
                "minibatch_size": 256,
                "epochs": 10,

                ## PIME
                "tracked_point_name": 'x2',
                "episodes_per_sample": 5,
            },
            "PID_and_Env": {
                "integrator_bounds": (-25, 25),
                "action_bounds": (0, 1),            # [V]
                "observation_max_bounds": (10, 10), # [cm]
                "PID_type": "PI",
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
                "dt": ("constant", {"constant": 2}),                 # dt sample time [s]
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
                "gamma": 0.99,
                "horizon": 200,
                "adam_stepsize": 3e-4,
                "minibatch_size": 256,
                "epochs": 10,


                ## PIME
                "tracked_point_name": 'x3',
                "episodes_per_sample": 5,
            },
            "PID_and_Env": {
                "integrator_bounds": (-25, 25),
                "action_bounds": (0, 100),        # [cmH2O]
                "PID_type": "PI",
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

create_env_function, logs_folder_path, hyperparameters = experiments['double_water_tank'].values()
env, scheduller, ensemble, trained_pid, pid_optimized_params = create_env_function(
    hyperparameters["seed"],
    hyperparameters["set_points"],
    hyperparameters["intervals"],
    hyperparameters["distributions"],
    hyperparameters["PID_and_Env"]["integrator_bounds"],
)
save_hyperparameters_as_json(env, 
                             logs_folder_path, 
                             hyperparameters
                             )

pime_ppo_controller = PIME_PPO(
                            env, 
                            scheduller, 
                            ensemble, 
                            # trained_pid,
                            # **pid_optimized_params,
                            optimized_Kp=0.9,
                            optimized_Ki=0.05,
                            optimized_Kd=0,
                            logs_folder_path=logs_folder_path,

                            # hyperparameters
                            **hyperparameters["PIME_PPO"],
                            integrator_bounds=hyperparameters["PID_and_Env"]["integrator_bounds"],
                            pid_type=hyperparameters["PID_and_Env"]["PID_type"],
                            sample_period=hyperparameters["distributions"]["dt"][1]["constant"],
                            seed=hyperparameters["seed"]
                            )

pime_ppo_controller.train(steps_to_run = 1_000_000)
