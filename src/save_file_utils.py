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



def create_textfile_if_not_exists(file_path: str, content: str) -> None:
    """
    Prints a warn if the file was created.
    """

    assert file_path.endswith(".txt"), "O file_path deve ser um arquivo .txt"

    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"[Aviso] Arquivo '{file_path}' foi criado.")
    

def read_textfile_and_increment_id(file_path: str) -> int:
    """
    Reads the content of a text file, increments it by 1 and saves it back to the file.
    """
    with open(file_path, 'r') as f:
        content = f.read()

    content = int(content) + 1

    with open(file_path, 'w') as f:
        f.write(str(content))

    return content
