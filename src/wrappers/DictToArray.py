import gymnasium
from gymnasium import spaces
import numpy as np


# ESSE WRAPPER NÃO ESTÁ SENDO UTILIZADO NO MOMENTO



# def map_name_to_array_index(obs: spaces.Dict) -> dict:
#     _map = {}
#     for i, key in enumerate(obs.keys()):
#         _map[key] = i
#     return _map

# class DictToArrayWrapper(gymnasium.ObservationWrapper):
    
#     def __init__(self, env):
#         super(DictToArrayWrapper, self).__init__(env)

#         self.observation_space = spaces.Box(
#             low=-np.inf, 
#             high=np.inf, 
#             shape=(len(env.observation_space.spaces),), 
#             dtype=np.float32
#         )
#         # self.counter = 0

#     def observation(self, observation) -> np.ndarray:
#         # self.my_map = map_name_to_array_index(observation) # Gambiarra
#         # print(f"@@@@@@@@@@@@@@@@ [DictToArray.observation] {self.counter}")
#         # print(observation)
#         # print(list(observation.values()))
#         # print(np.array(list(observation.values()), dtype=np.float32))
#         # self.counter += 1
#         # print("@@@@@@@@@@@@@@@@")
#         obs_array = np.array(list(observation.values()), dtype=np.float32)
#         return obs_array
