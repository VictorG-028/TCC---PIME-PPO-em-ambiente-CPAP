import gymnasium
from gymnasium.spaces import Box, Dict as Gymnasium_Space_Dict
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_device

import torch as pyTorch
import torch.nn.functional as F
import torch.nn as nn

from enums.TerminationRule import TerminationRule
from enums.ErrorFormula import ErrorFormula
from modules.Ensemble import Ensemble
from modules.Scheduller import Scheduller

################################################################################

"""
Funções auxiliares que permitem o ambiente ter uma observação em Dict e 
ao mesmo tempo permite que o stable baseline receba uma observação em box
"""

# def flatten_observation_space(observation_space: Gymnasium_Space_Dict) -> Box:
#     """
#     Flatten a gymnasium Dict observation space to a Box observation space.
#     """
#     sizes = []
#     for space in observation_space.spaces.values():
#         if isinstance(space, Box):
#             sizes.append(np.prod(space.shape))
#         else:
#             raise NotImplementedError("Only Box spaces are supported in Dict observation space.")
    
#     flattened_size = sum(sizes)
#     return Box(low=-np.inf, high=np.inf, shape=(flattened_size,), dtype=np.float32)

# def flatten_observation(observation: dict) -> np.ndarray:
#     """
#     Flatten a dictionary observation to a numpy array.
#     """
#     flat_obs = []
#     for key, value in observation.items():
#         flat_obs.append(np.asarray(value).flatten())
#     return np.concatenate(flat_obs)

################################################################################

class CustomMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CustomMLP, self).__init__()
        
        self.fc1_upper_part = nn.Linear(input_dim -1, 6)
        self.fc1_lower_part = nn.Linear(1, 2)

        self.fc2_upper_part = nn.Linear(6, 6)
        self.fc2_lower_part = nn.Linear(2, 2)

        self.fc3_upper_part = nn.Linear(6, 6)
        self.fc3_lower_part = nn.Linear(2, 2)

        self.fc4_merge_upper_and_lower = nn.Linear(input_dim, 6)
        self.fc5 = nn.Linear(6, output_dim)

    def forward(self, x):
        x_values_and_yRef_t = x[:, :-1] # Todos menos o último
        z_t = x[:, -1].unsqueeze(1)     # Última posição e ajuste de dimensão

        upper_part = F.relu(self.fc1_upper_part(x_values_and_yRef_t))
        lower_part = F.relu(self.fc1_lower_part(z_t))

        upper_part = F.relu(self.fc2_upper_part(upper_part))
        lower_part = F.relu(self.fc2_lower_part(lower_part))

        upper_part = F.relu(self.fc3_upper_part(upper_part))
        lower_part = F.relu(self.fc3_lower_part(lower_part))

        merged = pyTorch.cat((upper_part, lower_part), dim=1) # Junta em um único tensor
        merged = F.relu(self.fc4_merge_upper_and_lower(merged))

        return self.fc5(merged)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(*args, **kwargs)

        input_dim = self.features_extractor.features_dim
        self.mlp_extractor = CustomMLP(input_dim, 1) # <---

        # Redefina as camadas de ação e valor
        # TODO: validar as redes neurais de ação e valor
        self.action_net = nn.Linear(6, self.action_space.shape[0]) # Faz Gymnaisum.Space.Box com shape (1,) retornar 1
        self.value_net = nn.Linear(6, 1)

    def forward(self, obs):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        return self.action_net(latent_pi), self.value_net(latent_vf)
    
    # def obs_to_tensor(self, observation):
    #     if isinstance(observation, dict):
    #         for key, value in observation.items():
    #             if not isinstance(value, np.ndarray):
    #                 observation[key] = np.array(value)
    #         obs_tensor = {key: pyTorch.tensor(value, dtype=pyTorch.float32) for key, value in observation.items()}
    #     else:
    #         obs_tensor = pyTorch.tensor(observation, dtype=pyTorch.float32)
        
    #     return obs_tensor, False


class PIController:
    """ A PID controller that don't uses the D (derivative) component and uses 
    a scheduller for multi-setpoint control
    """
    def __init__(self, 
                 kp: np.float64, 
                 ki: np.float64, 
                 scheduller: Scheduller
                ) -> None:
        self.kp = kp
        self.ki = ki
        self.integral = 0

        self.timstep = 0
        self.scheduller = scheduller
        self.set_point = scheduller.get_set_point_at(step=0)


    def control(self, observation, timestep):
        set_point = self.scheduller.get_set_point_at(step=timestep)

        # TODO: Ver se esse if que reseta a integral deve existir
        hasChanged = self.set_point != set_point
        if (hasChanged):
            self.integral = 0
            self.set_point = set_point

        self.timstep += 1

        error = np.abs(set_point - observation)
        self.integral += error
        action = self.kp * error + self.ki * self.integral
        return action
    

    def reset(self):
        self.timstep = 0


class PIME_PPO:
    """An algorithm that uses PID and PPO."""

    def __init__(self, 
                 env: gymnasium.Env,
                 scheduller: Scheduller,
                 ensemble: Ensemble,
                 verbose: int = 1,
                 kp: np.float64 = 0.1, 
                 ki: np.float64 = 0.1,
                 tracked_point_name: str = 'x1',
                 buffer_size: int = 2048 # Possible acurate buffer_size = ensemble.size * episode_lenght
                 ) -> None:
        self.env = env
        self.sheduller = scheduller
        self.ensemble = ensemble
        self.ppo = PPO(CustomActorCriticPolicy, env, verbose=verbose) # "MlpPolicy"
        self.pi_controller = PIController(kp, ki, scheduller) # TODO: receber um PID treinado
        self.tracked_point_name = tracked_point_name

        # flat_obs_space = flatten_observation_space(self.env.observation_space)
        # print(buffer_size)
        # print(self.env.observation_space)
        # print(self.env.action_space)
        # print(flat_obs_space, type(flat_obs_space))
        # print("@@@@@@@@@@@@@@@@@@")
        self.rollout_buffer = RolloutBuffer(buffer_size,
                                            self.env.observation_space,
                                            self.env.action_space,
                                            gamma=0.99,
                                            gae_lambda=0.97,
                                            n_envs=1)
        # print("FIM")
        # exit()


    def train(self, n_episodes = 10) -> None:
        buffer = []
        for episode in range(n_episodes):
            obs, truncated = self.env.reset()
            done = False
            i = 0
            for ensemble_parameters in self.ensemble.get_all_params_set():
                self.env.unwrapped.set_ensemble_params(ensemble_parameters)
                
                while not done:
                    # flattened_obs = flatten_observation(obs)
                    # flattened_obs = np.array([obs[key] for key in obs.keys()])
                    # flattened_obs = flattened_obs.reshape((1, -1))  # Garantir formato bidimensional
                    # assert isinstance(flattened_obs, np.ndarray), "flattened_obs must be a np.ndarray"
                    # print("[PIME_PPO.py] returned obs: ", obs)
                    print(f"[PIME_PPO.py] returned wrapped_obs: {obs} {type(obs)} shape: {obs.shape}")
                    # print(f"[PIME_PPO.py] returned flattened_obs: {flattened_obs} {type(flattened_obs)} shape: {flattened_obs.shape}")
                    index = self.env.my_map[self.tracked_point_name]
                    input(">>>")
                    pi_action = self.pi_controller.control(
                        index, # obs[self.tracked_point_name], # Não é mais um dicionário
                        self.env.unwrapped.timestep
                    )

                    ppo_action, _ = self.ppo.predict(obs) # <--------------

                    action = pi_action + ppo_action # ppo_action[0]

                    # TODO: estudar e implementar discretização com método de Euler (referência 32 do paper)
                    # action = np.clip(int(action), 0, env.action_space.n - 1)

                    next_obs, reward, done, truncated = self.env.step(action)
                    # flattened_next_obs = flatten_observation(next_obs)
                    buffer.append((obs, obs['y_ref'], action, next_obs, reward))
                    
                    # TODO: rede neural customizada + ajustar buffer conforme rede
                    self.rollout_buffer.add(flattened_obs, action, flattened_next_obs, reward, done)
                    
                    obs = next_obs # Can update obs after storing both in buffer

            self.ppo.learn(self.rollout_buffer, total_timesteps=self.rollout_buffer.buffer_size)
            self.rollout_buffer.reset()
            obs, _ = self.env.reset()
            done = False
