



from datetime import datetime

import pandas as pd

from algorithms.PID_Controller import PIDController
from save_file_utils import create_dir_if_not_exists, save_hyperparameters_as_json


def run_PID_only_episode(experiment: dict[str, any], should_save_records = False) -> None:

    logs_folder_path = experiment["training_logs_folder_path"]
    create_env_function = experiment["create_env_function"] 
    hyperparameters = experiment["hyperparameters"]

    current_date_time = datetime.now().strftime("%d-%m-%H%M")  # Format: day-month-hourminute
    logs_folder_path = logs_folder_path.format(
        category="PID_only",
        id=current_date_time
    )

    env, scheduller, ensemble, trained_pid, pid_optimized_params = create_env_function(
        hyperparameters["seed"],
        hyperparameters["set_points"],
        hyperparameters["intervals"],
        hyperparameters["distributions"],
        **hyperparameters["PID_and_Env"]
    )

    dict_to_save = {
        "category": "PID_only",
        "Kp": hyperparameters["PIME"]["Kp"],
        "Ki": hyperparameters["PIME"]["Ki"],
        "Kd": hyperparameters["PIME"]["Kd"],
        "integrator_bounds": hyperparameters["PID_and_Env"]["integrator_bounds"],
        "pid_type": hyperparameters["PID_and_Env"]["pid_type"],
        "create_env_function": experiment["create_env_function"],
        "hyperparameters": {}
    }
    save_hyperparameters_as_json(
        env=env, 
        logs_folder_path=logs_folder_path, 
        params_dict=dict_to_save
    )

    controller = PIDController(
        Kp = hyperparameters["PIME"]["Kp"],
        Ki = hyperparameters["PIME"]["Ki"],
        Kd = hyperparameters["PIME"]["Kd"],
        dt = hyperparameters["distributions"]["dt"][1]["constant"],
        integrator_bounds = hyperparameters["PID_and_Env"]["integrator_bounds"],
        controller_type = hyperparameters["PID_and_Env"]["pid_type"]
    )

    records = [] # (*x_vector, y_ref, z_t, PID_action, reward, error, steps_in_episode)
    returns = 0

    done = False
    steps_in_episode = 0
    sample_parameters = ensemble.generate_sample()
    obs, truncated = env.reset(options = {"ensemble_sample_parameters": sample_parameters})

    while not done:

        action = controller(env.unwrapped.error)

        next_obs, reward, done, truncated, info = env.step(action)
        steps_in_episode += 1
        returns += reward

        records.append((*next_obs, action, reward, env.unwrapped.error, steps_in_episode, returns))

        obs = next_obs # Can update obs after storing in buffer

    if (should_save_records):
        pd.DataFrame(
            records, 
            columns=[f"x{i+1}" for i in range(env.unwrapped.x_size)] + \
                    ["y_ref", "z_t", "PID_action", "reward", "error", "steps_in_episode", "returns"]
        ).to_csv(
            f"{logs_folder_path}/records.csv", 
            index=False
        )

        # save last used sample_parameters
        pd.DataFrame(
            sample_parameters.items(), 
            columns=["parameter_name", "parameter_value"]
        ).to_json(
            f"{logs_folder_path}/last_sample_parameters.json", 
            orient="records"
        )

