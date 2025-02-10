

from algorithms.PIME_PPO import PIME_PPO
from SaveFiles import save_hyperparameters_as_json


def run_training(experiment: dict[str, any]) -> None:

    create_env_function, logs_folder_path, hyperparameters = experiment.values()
    env, scheduller, ensemble, trained_pid, pid_optimized_params = create_env_function(
        hyperparameters["seed"],
        hyperparameters["set_points"],
        hyperparameters["intervals"],
        hyperparameters["distributions"],
        **hyperparameters["PID_and_Env"]
    )
    # hyperparameters["PIME_PPO"]["Kp"] = pid_optimized_params["optimized_Kp"],
    # hyperparameters["PIME_PPO"]["Ki"] = pid_optimized_params["optimized_Ki"],
    # hyperparameters["PIME_PPO"]["Kd"] = pid_optimized_params["optimized_Kd"],

    save_hyperparameters_as_json(env, 
                                 logs_folder_path, 
                                 experiment
                                )

    pime_ppo_controller = PIME_PPO(
                                env, 
                                scheduller, 
                                ensemble, 
                                **hyperparameters["PIME_PPO"],
                                logs_folder_path=logs_folder_path,
                                integrator_bounds=hyperparameters["PID_and_Env"]["integrator_bounds"],
                                pid_type=hyperparameters["PID_and_Env"]["pid_type"],
                                sample_period=hyperparameters["distributions"]["dt"][1]["constant"],
                                seed=hyperparameters["seed"]
                                )

    pime_ppo_controller.train(steps_to_run = 1_000_000, 
                            extra_record_only_pid=True,
                            should_save_records=True
                            )
