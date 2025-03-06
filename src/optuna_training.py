from datetime import datetime
import json
import os
from time import sleep

import optuna
from functools import partial

from algorithms.PIME_PPO import PIME_PPO
from save_file_utils import create_dir_if_not_exists, create_textfile_if_not_exists, read_textfile_and_increment_id, save_hyperparameters_as_json

def run_optuna(
        experiment: dict[str, any], 
        steps_to_run = 100_000,
        extra_record_only_pid = False,
        should_save_records = False,
        should_save_trained_model = False,
        n_trials = 1, 
        n_jobs = 1, 
        use_GPU = False
    ) -> None:

    def objective(trial: optuna.Trial, experiment: dict[str, any]) -> float:
        """
        This function represents a thread inside optuna study.
        """

        def set_hyperparameter_search_space(trial: optuna.Trial, 
                                            base_hyperparameters: dict[str, any]
                                            ) -> dict[str, any]:
            
            global search_space
            
            # water tank
            # search_space = {
            #     "Kp": trial.suggest_float("Kp", 0.1, 10.0, step=0.1),   # 100 opções
            #     "Ki": trial.suggest_float("Ki", 0.001, 0.100, step=0.001), # 100 opções
            #     "ensemble_size": 1,
            #     "episodes_per_sample": 1
            # }
            # search_space = {
            #     "Kp": 8.8,
            #     "Ki": 11.46,
            #     "ensemble_size": 1,
            #     "episodes_per_sample": 1
            # }
            # search_space = {
            #     "vf_coef": trial.suggest_float("vf_coef", 0.01, 10.0, log=True),
            #     "horizon": trial.suggest_int("horizon", 200, 2000, step=200),
            #     "minibatch_size": trial.suggest_categorical("minibatch_size", [32, 64, 128, 256]),
            #     "adam_stepsize": trial.suggest_float("adam_stepsize", 1e-5, 3e-4, log=True),
            #     "neurons_per_layer": trial.suggest_int("neurons_per_layer", 6, 60, step=6),
            #     "divide_neural_network": trial.suggest_categorical("divide_neural_network", [True, False]),
            #     "activation_function_name": trial.suggest_categorical("activation_function_name", ["no activation", "relu", "tanh"]),
            #     "ensemble_size": 1,
            # }

            # CPAP
            # search_space = {
            #     "Kp": trial.suggest_float("Kp", 0.1, 10.0, step=0.1),   # 100 opções
            #     "Ki": trial.suggest_float("Ki", 0.01, 10.0, step=0.01), # 1000 opções
            #     "ensemble_size": 1,
            #     "episodes_per_sample": 1
            # }
            search_space = {
                "vf_coef": trial.suggest_float("vf_coef", 0.01, 10.0, log=True),
                "horizon": trial.suggest_int("horizon", 200, 2000, step=200),
                "minibatch_size": trial.suggest_categorical("minibatch_size", [32, 64, 128, 256]),
                "adam_stepsize": trial.suggest_float("adam_stepsize", 1e-5, 3e-4, log=True),
                "neurons_per_layer": trial.suggest_int("neurons_per_layer", 6, 60, step=6),
                "divide_neural_network": trial.suggest_categorical("divide_neural_network", [True, False]),
                "activation_function_name": trial.suggest_categorical("activation_function_name", ["no activation", "relu", "tanh"]),
                "ensemble_size": 1,
                "episodes_per_sample": 5
            }

            # Atualiza os hiperparâmetros do experimento com os espaços de possíveis valores
            new_hyperparameters = base_hyperparameters.copy()
            new_hyperparameters["PIME_PPO"].update(search_space)
            
            return new_hyperparameters

        hyperparameters = set_hyperparameter_search_space(trial, experiment["hyperparameters"])

        env, scheduller, ensemble, trained_pid, pid_optimized_params = experiment['create_env_function'](
            hyperparameters["seed"],
            hyperparameters["set_points"],
            hyperparameters["intervals"],
            hyperparameters["distributions"],
            **hyperparameters["PID_and_Env"]
        )

        current_date_time = datetime.now().strftime("%d-%m-%H%M")  # Format: day-month-hourminute
        trial_logs_folder_path = experiment['training_logs_folder_path'].format(
                                    id=current_date_time
                                ) + f"-optuna-trial_{trial.number}"

        save_hyperparameters_as_json(
            env, 
            trial_logs_folder_path, 
            experiment
        )

        pime_ppo_controller = PIME_PPO(
            env, scheduller, ensemble,
            **hyperparameters["PIME_PPO"],
            use_GPU=use_GPU,
            logs_folder_path=trial_logs_folder_path,
            integrator_bounds=hyperparameters["PID_and_Env"]["integrator_bounds"],
            pid_type=hyperparameters["PID_and_Env"]["pid_type"],
            sample_period=hyperparameters["distributions"]["dt"][1]["constant"],
            seed=hyperparameters["seed"]
        )

        score = pime_ppo_controller.train(
            steps_to_run=steps_to_run,
            extra_record_only_pid=extra_record_only_pid,
            should_save_records=should_save_records,
            should_save_trained_model=should_save_trained_model,
        )

        return score


    study = optuna.create_study(direction="maximize")
    objective_with_experiment = partial(objective, experiment=experiment)
    study.optimize(objective_with_experiment, n_trials=n_trials, n_jobs=n_jobs) # n_trials=17*3*7=357 jobs=17

    print(f"Melhores hiperparâmetros encontrados: {study.best_params}")

    # Load all directories names inside trial_logs_folder_path
    base_dir = experiment["training_logs_folder_path"].format(id="") # logs/ppo/CPAP/trainings/{id}

    best_folder_name = None
    for f in os.listdir(base_dir):
        if os.path.isdir(f"{base_dir}/{f}") and f"_{study.best_trial.number}" in f:
            best_folder_name = f
            break
    assert best_folder_name is not None, f"Folder with best trial number {study.best_trial.number} not found."

    # Rename directory for better identification
    os.rename(base_dir + best_folder_name, 
              base_dir + best_folder_name + "_best")
    

    # Adds more info to be saved
    study.best_params["optuna_hyperparams"] = {
        "experiment_name": experiment["training_logs_folder_path"].split("/")[2], # CPAP or double_water_tank
        "steps_to_run": steps_to_run,
        "should_save_records": should_save_records,
        "extra_record_only_pid": extra_record_only_pid,
        "should_save_trained_model": should_save_trained_model,
        "n_trials": n_trials,
        "n_jobs": n_jobs,
        "use_GPU": use_GPU
    }
    study.best_params["optuna_search_space"] = search_space

    # save best parameters
    create_dir_if_not_exists(experiment['best_results_folder_path'])
    create_textfile_if_not_exists(experiment['best_results_folder_path'] + "last_experiment_id.txt", content="0")
    next_id = read_textfile_and_increment_id(experiment['best_results_folder_path'] + "last_experiment_id.txt")
    with open(experiment['best_results_folder_path'] + f"study_{next_id}_best_hyperparameters.json", "w") as f:
        json.dump(study.best_params, f, indent=4)
