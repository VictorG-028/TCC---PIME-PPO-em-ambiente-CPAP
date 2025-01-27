from typing import Any, Optional
import gymnasium
from gymnasium import spaces
import numpy as np



def map_name_to_array_index(obs: spaces.Dict) -> dict:
    _map = {}
    for i, key in enumerate(obs.keys()):
        _map[key] = i # key = name -> i == array_index
    return _map

class DictToArrayWrapper(gymnasium.ObservationWrapper):
    
    def __init__(self, env):
        super(DictToArrayWrapper, self).__init__(env)

        self.observation_space = spaces.Box(
            low=-float('inf'), 
            high=float('inf'), 
            shape=(len(env.observation_space.spaces),), 
            dtype=np.float32
        )
        # self.counter = 0

    def observation(self, observation) -> np.ndarray:
        # self.my_map = map_name_to_array_index(observation) # Gambiarra
        # print(f"@@@@@@@@@@@@@@@@ [DictToArray.observation] {self.counter}")
        # print(observation)
        # print(list(observation.values()))
        # print(np.array(list(observation.values()), dtype=np.float32))
        # self.counter += 1
        # input(">>>")
        # print("@@@@@@@@@@@@@@@@")
        obs_array = np.array(list(observation.values()), dtype=np.float32)

        return obs_array
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None) -> tuple[np.ndarray, dict]:
        obs, _ = self.env.reset(seed=seed, options=options)
        obs: dict[str, np.float64]

        array_obs = np.array(list(obs.values()), dtype=np.float64)
        # print(f"{obs=}")
        # print(list(obs.values()))
        # print(array_obs)
        # print(array_obs.shape)
        # print(array_obs.reshape((-1,)))
        # print(array_obs.shape)
        # input(">3>")
        
        return array_obs, _
