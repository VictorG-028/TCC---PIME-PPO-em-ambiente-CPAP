from datetime import datetime

from algorithms.PIME_PPO import PIME_PPO
from save_file_utils import save_hyperparameters_as_json


def run_training(
        experiment: dict[str, any], 
        steps_to_run = 100_000,
        extra_record_only_pid = False,
        should_save_records = False,
        should_save_trained_model = False,
        use_GPU = False
    ) -> None:

    current_date_time = datetime.now().strftime("%d-%m-%H%M")  # Format: day-month-hourminute
    training_logs_folder_path = experiment["training_logs_folder_path"].format(id=current_date_time)
    create_env_function = experiment["create_env_function"] 
    hyperparameters = experiment["hyperparameters"]

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

    save_hyperparameters_as_json(
        env, 
        training_logs_folder_path, 
        experiment
    )

    pime_ppo_controller = PIME_PPO(
        env, 
        scheduller, 
        ensemble, 
        **hyperparameters["PIME_PPO"],
        use_GPU=use_GPU,
        logs_folder_path=training_logs_folder_path,
        integrator_bounds=hyperparameters["PID_and_Env"]["integrator_bounds"],
        pid_type=hyperparameters["PID_and_Env"]["pid_type"],
        sample_period=hyperparameters["distributions"]["dt"][1]["constant"],
        seed=hyperparameters["seed"]
    )

    score = pime_ppo_controller.train(
        steps_to_run=steps_to_run, 
        extra_record_only_pid=extra_record_only_pid,
        should_save_records=should_save_records,
        should_save_trained_model=should_save_trained_model
    )
