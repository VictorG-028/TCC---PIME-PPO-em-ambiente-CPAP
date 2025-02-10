import json
import os
from typing import Any
import gymnasium


def save_hyperparameters_as_json(env: gymnasium.Env, 
                                 logs_folder_path: str,
                                 params_dict: dict[str, Any]
                                 ) -> None:
    """
    Adds info from env to hypeparameters and then saves hypeparameters as a json file.
    """
    
    # hyperparameters["action_space"] = {
    #     "min": str(env.unwrapped.action_space.low.tolist()[0]),
    #     "max": str(env.unwrapped.action_space.high.tolist()[0])
    # }
    params_copy = params_dict.copy()
    params_copy.pop("create_env_function")

    params_copy["hyperparameters"]["observation_space"] = {}
    for key, box in env.unwrapped.observation_space.spaces.items():
        params_copy["hyperparameters"]["observation_space"][key] = {
            "min": str(box.low.tolist()[0]), 
            "max": str(box.high.tolist()[0])
        }

    create_dir_if_not_exists(logs_folder_path)
    with open(f"{logs_folder_path}/hyperparameters.json", "w") as f:
        json.dump(params_copy, f, indent=4)



def create_dir_if_not_exists(dir_path: str) -> None:
    """
    Prints a warn if the directory was created.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"[Aviso] Pasta '{dir_path}' foi criada para armazenar os logs.")
