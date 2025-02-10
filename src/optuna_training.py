import optuna
from functools import partial

from algorithms.PIME_PPO import PIME_PPO
from SaveFiles import create_dir_if_not_exists, save_hyperparameters_as_json

def run_optuna(experiment: dict[str, any]) -> None:

    def objective(trial: optuna.Trial, experiment: dict[str, any]) -> float:


        def set_hyperparameter_search_space(trial: optuna.Trial, 
                                            base_hyperparameters: dict[str, any]
                                            ) -> dict[str, any]:
            optimized_hyperparameters = {
                "vf_coef": trial.suggest_float("vf_coef", 0.01, 10.0, log=True),
                "horizon": trial.suggest_int("horizon", 200, 2000),
                "minibatch_size": trial.suggest_categorical("minibatch_size", [32, 64, 128, 256]),
                "adam_stepsize": trial.suggest_float("adam_stepsize", 1e-5, 3e-4, log=True),
                "neurons_per_layer": trial.suggest_int("neurons_per_layer", 6, 100),
                "divide_neural_network": trial.suggest_categorical("divide_neural_network", [True, False]),
                "activation_function_name": trial.suggest_categorical("activation_function_name", ["no activation", "relu", "tanh"]),
                "ensemble_size": 1,
            }

            # Atualiza os hiperparâmetros do experimento com os espaços de possíveis valores
            new_hyperparameters = base_hyperparameters.copy()
            new_hyperparameters["PIME_PPO"].update(optimized_hyperparameters)
            
            return new_hyperparameters

        
        base_hyperparameters = experiment["hyperparameters"]
        hyperparameters = set_hyperparameter_search_space(trial, base_hyperparameters)

        env, scheduller, ensemble, trained_pid, pid_optimized_params = experiment['create_env_function'](
            hyperparameters["seed"],
            hyperparameters["set_points"],
            hyperparameters["intervals"],
            hyperparameters["distributions"],
            **hyperparameters["PID_and_Env"]
        )

        save_hyperparameters_as_json(env, 
                                    experiment['logs_folder_path'], 
                                    experiment
                                    )

        pime_ppo_controller = PIME_PPO(
            env, scheduller, ensemble,
            **hyperparameters["PIME_PPO"],
            logs_folder_path=experiment["logs_folder_path"],
            integrator_bounds=hyperparameters["PID_and_Env"]["integrator_bounds"],
            pid_type=hyperparameters["PID_and_Env"]["pid_type"],
            sample_period=hyperparameters["distributions"]["dt"][1]["constant"],
            seed=hyperparameters["seed"]
        )

        score = pime_ppo_controller.train(steps_to_run = 1_000_000,
                                          extra_record_only_pid=True,
                                          should_save_records=True
                                        )

        return score


    study = optuna.create_study(direction="maximize")
    objective_with_experiment = partial(objective, experiment=experiment)
    study.optimize(objective_with_experiment, n_trials=17*3*7, n_jobs=17)

    print("Melhores hiperparâmetros encontrados:")
    print(study.best_params)

    # save best parameters
    create_dir_if_not_exists(experiment['logs_folder_path'])
    with open(experiment['logs_folder_path'] + "/best_hyperparameters.json", "w") as f:
        f.write(str(study.best_params))
