from datetime import datetime
from algorithms.PIME_PPO import PIME_PPO
from algorithms.PIME_TD3 import PIME_TD3
from save_file_utils import save_hyperparameters_as_json

def run_training(
        algorithm_class: PIME_PPO | PIME_TD3,
        experiment: dict[str, any], 
        category: str,
        steps_to_run = 100_000,
        extra_record_only_pid = False,
        should_save_records = False,
        should_save_trained_model = False,
        extra_record_trained_agent = False,
        use_GPU = False
    ) -> None:


    training_logs_folder_path = experiment["training_logs_folder_path"]
    create_env_function = experiment["create_env_function"] 
    hyperparameters = experiment["hyperparameters"]

    current_date_time = datetime.now().strftime("%d-%m-%H%M")  # Format: day-month-hourminute
    training_logs_folder_path = training_logs_folder_path.format(
        category=category, 
        id=current_date_time
    )

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

    controller = algorithm_class(
        env,
        scheduller,
        ensemble,
        **hyperparameters["PIME"],
        use_GPU=use_GPU,
        logs_folder_path=training_logs_folder_path,
        integrator_bounds=hyperparameters["PID_and_Env"]["integrator_bounds"],
        agent_action_bounds=hyperparameters["PID_and_Env"]["agent_action_bounds"],
        pid_type=hyperparameters["PID_and_Env"]["pid_type"],
        sample_period=hyperparameters["distributions"]["dt"][1]["constant"],
        seed=hyperparameters["seed"],
    )

    print(f"Start {algorithm_class.__class__.__repr__} training with {steps_to_run} steps")
    score = controller.train(
        steps_to_run=steps_to_run, 
        extra_record_only_pid=extra_record_only_pid,
        should_save_records=should_save_records,
        should_save_trained_model=should_save_trained_model,
        extra_record_only_agent=extra_record_trained_agent
    )
    print(f"End {algorithm_class.__repr__} training with {steps_to_run} steps")
