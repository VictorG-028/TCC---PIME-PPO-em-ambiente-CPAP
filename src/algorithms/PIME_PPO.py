from typing import Callable, Literal, Optional
import math
import re
from datetime import datetime
import locale
import gymnasium
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.logger import configure

import torch as pyTorch
import torch.nn.functional as torch_nn_functional
import torch.nn as nn

from algorithms.PID_Controller import PIDController
from enums.TerminationRule import TerminationRule
from enums.ErrorFormula import ErrorFormula
from modules.EnsembleGenerator import EnsembleGenerator
from modules.SaveFiles import create_dir_if_not_exists
from modules.Scheduller import Scheduller



class CustomMLP(nn.Module):
    def __init__(self, 
                 feature_dim: int,
                 last_layer_dim_pi: int = 64,
                 last_layer_dim_vf: int = 64
                 ):
        super().__init__()
        
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create the same network arquitecture for policy and value networks
        self.policy_upper_net, self.policy_lower_net, self.policy_merged_net = self.create_networks(feature_dim, last_layer_dim_pi)
        self.value_upper_net, self.value_lower_net, self.value_merged_net = self.create_networks(feature_dim, last_layer_dim_vf)
    

    def create_networks(self, feature_dim, last_layer_dim):
        upper_net = nn.Sequential(
            nn.Linear(feature_dim - 1, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 4), # Finish with 4
            nn.ReLU()
        )

        lower_net = nn.Sequential(
            nn.Linear(1, 2),
            nn.ReLU(),
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 2), # Finish with 2
            nn.ReLU()
        )

        merged_net = nn.Sequential(
            nn.Linear(4+2, 6), # Recives 4 + 2 from both previous finished layers
            nn.ReLU(),
            nn.Linear(6, last_layer_dim),
            nn.ReLU()
        )

        return upper_net, lower_net, merged_net


    def forward(self, input_vector: pyTorch.Tensor):
        # print(f"[CustomMLP forward] {input_vector=}")
        return self.forward_actor(input_vector), self.forward_critic(input_vector)


    def forward_actor(self, input_vector: pyTorch.Tensor) -> pyTorch.Tensor:
        # print(f"[CustomMLP forward_actor] {input_vector=}")
        
        # Split input_vector
        x_values_and_yRef_t = input_vector[:, :-1] # Todos menos o último
        z_t = input_vector[:, -1].unsqueeze(1)     # Última posição e ajuste de dimensão

        # Policy network
        upper_tensor = self.policy_upper_net(x_values_and_yRef_t)
        lower_tensor = self.policy_lower_net(z_t)
        merged_tensor = pyTorch.cat((upper_tensor, lower_tensor), dim=1)
        return self.policy_merged_net(merged_tensor)


    def forward_critic(self, input_vector: pyTorch.Tensor) -> pyTorch.Tensor:
        # print(f"[CustomMLP forward_critic] {input_vector=}")

        # Split input_vector
        x_values_and_yRef_t = input_vector[:, :-1] # Todos menos o último
        z_t = input_vector[:, -1].unsqueeze(1)     # Última posição e ajuste de dimensão

        # Value network
        upper_tensor = self.value_upper_net(x_values_and_yRef_t)
        lower_tensor = self.value_lower_net(z_t)
        merged_tensor = pyTorch.cat((upper_tensor, lower_tensor), dim=1)
        return self.value_merged_net(merged_tensor)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, 
                 observation_space: gymnasium.spaces.Space,
                 action_space: gymnasium.spaces.Space,
                 lr_schedule: Callable[[float], float],
                 *args,
                 **kwargs,
                 ):
        kwargs["ortho_init"] = False # Disable orthogonal initialization

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomMLP(
            self.features_dim, 
            self.action_space.shape[0], 
            self.action_space.shape[0]
        )





class PIME_PPO:
    """
    PIME = Prior_PI_controller & Integrator & Model Ensemble
    
    [en]
    An algorithm that uses PID and PPO.
    
    [pt-br]
    Um algoritmo que usa um controlador PID e o algoritmo de aprendizado por reforço PPO.
    """

    def __init__(self, 
                 env: gymnasium.Env,                                        # gymnasium.Env like set point env
                 scheduller: Scheduller,                                    # manage the set_point at each step
                 ensemble_generator: EnsembleGenerator,                     # manage the random inicialization of parameters
                 ensemble_size: int = 2,                                    # number of models in ensemble
                 pid_controller: Optional[Callable[[float], float]] = None, # Tuned pid encapsulated inside a function
                 optimized_Kp: np.float64 = 1,                              # 
                 optimized_Ki: np.float64 = 1,                              # 
                 optimized_Kd: np.float64 = 0,                              # Kd is not used in PIME PPO
                 tracked_point_name: str = 'x1',                            # string like "x[integer]"
                 verbose: int = 1,                                          # 
                 logs_folder_path: str = "logs/ppo/",                       # PPO logs
                 buffer_size: Optional[int] = None,                         # buffer_size = ensemble_size * episode_lenght (minimum size needed to not get buffer overflow error)
                 episodes_per_sample: int = 5,                              # Number of episodes collected for one parameter set
                 gamma = 0.99,                                              # PPO param - Discount factor
                 clip_range: float = 0.2,                                   # PPO param
                 gae_lambda: float = 0.97,                                  # PPO param
                 vf_coef: float = 1.0,                                      # PPO param c1 = vf_coef
                 ent_coef: float = 0.02,                                    # PPO param c2 = ent_coef
                 integrator_bounds: tuple[int, int] = (-25, 25),            # Clip para PID e PPO (formula do integrator)
                 pid_type: Literal["PID", "PI", "P"] = "PI",                # PID, PI or P
                 sample_period: int = 1,                                    # Euler method sampling period (dt)
                 seed: Optional[int] = None,                                # Seed for reproducibility
                 ) -> None:
        """
        #### NOTE:
        buffer size depende do tamanho do episódio e do tamanho do ensemble. 
        O tamanho do episódio depende da termination rule. 
        O tamanho do ensemble é definido pelo usuário.
        """
        assert env.unwrapped.error_formula is not None, \
            "Must have an error_formula (type ErrorFormula or Callable[y:float, y_ref:float], output:float]) propertie inside env."
        
        regex = r"x(?<!0)\d+" # (?<!0) rejects "x0", "x01", "x02" cases
        assert bool(re.match(regex, tracked_point_name)), f"tracked_point_name must be like 'x[integer]'. Ex: 'x1', 'x2', 'x3', etc."

        if pid_controller is not None:
            self.pid_controller = pid_controller
        else:
            self.pid_controller = PIDController(optimized_Kp, optimized_Ki, optimized_Kd, 
                                                integrator_bounds, 
                                                sample_period, 
                                                env.unwrapped.error_formula,
                                                controller_type = pid_type
                                                ) 
            
        self.env = env
        self.scheduller = scheduller
        self.ensemble = ensemble_generator
        self.ensemble_size = ensemble_size
        self.tracked_point_name = tracked_point_name
        self.episodes_per_sample = episodes_per_sample
        self.sample_period = sample_period
        self.logs_folder_path = logs_folder_path


        # Nota: Ao criar uma nova TerminationRule, é necessário fazer um novo elif que calcula o steps_per_episode em função dessa nova regra.
        if env.unwrapped.max_step is not None:
            self.steps_per_episode = env.unwrapped.max_step
        else:
            self.steps_per_episode = env.unwrapped.scheduller.intervals_sum


        if buffer_size is None:
            # n_envs = buffer size, uma vez que num_envs = 1
            buffer_size = self.steps_per_episode * ensemble_size * episodes_per_sample
                          
        self.buffer_size = buffer_size
            

        """Lembrar:
            Definir buffer_size usando rollout_buffer_kwargs ou buffer_size não funciona. 
            Olhando o código fonte, precisa definir n_steps que é  "the number 
            of experiences which is collected from a single environment under 
            the current policy before its next update".
        """
        self.ppo = PPO(CustomActorCriticPolicy, 
                       env, 
                       verbose               = verbose,              #
                       gamma                 = gamma,                # Discount factor
                       gae_lambda            = gae_lambda,           # Factor for trade-off of bias vs variance for Generalized Advantage Estimator. Equivalent to classic advantage when set to 1.
                       clip_range            = clip_range,           #
                       clip_range_vf         = None,                 # No clip applyed to vf
                       vf_coef               = vf_coef,              # 0.02
                       ent_coef              = ent_coef,             # 1.0
                       rollout_buffer_class  = RolloutBuffer,        #
                       seed                  = seed,                 #
                       # rollout_buffer_kwargs = {'buffer_size': buffer_size},
                       # buffer_size           = buffer_size,
                       n_steps               = self.buffer_size,
                       )
        
        # Setup logger (mandatory to avoid "AttributeError: 'PPO' object has no attribute '_logger'.")
        # https://stable-baselines3.readthedocs.io/en/master/common/logger.html
        create_dir_if_not_exists(logs_folder_path)
        new_logger = configure(logs_folder_path, ["stdout", "csv", "tensorboard"])
        self.ppo.set_logger(new_logger)




    def train(self, steps_to_run = 100_000) -> None:
        """
        steps_to_run [int]: Approximattely the ammount of calls to env.step function. The ammount of calls cannot be less than steps_to_run, but can be more if nedded to finish an iteration. Iteration is the ammount of times to loop through all models in ensemble. Each complete ensemble loop is 1 iteration.
        """

        assert steps_to_run > 0, "steps_to_run must be greater than 0."

        records = [] # (*x_vector, y_ref, z_t, PID_action, PPO_action, action, reward, error, steps_in_episode, is_start_episode)
        
        # steps_extra_info: list[tuple] = []
        steps_in_episode = 0
        total_steps_counter = 0

        steps_per_iteration = self.steps_per_episode * self.ensemble_size * self.episodes_per_sample
        iterations: int = math.ceil(steps_to_run / steps_per_iteration)
        print(f"Steps requested: {steps_to_run}, Steps to be executed: {iterations * steps_per_iteration} (Iterations: {iterations})")

        for iteration in range(1, iterations+1):

            for m in range(self.ensemble_size):
                
                for j in range(self.episodes_per_sample):
                    sample_parameters = self.ensemble.generate_sample()
                    obs, truncated = self.env.reset(options = {"ensemble_sample_parameters": sample_parameters})
                    is_start_episode = True # Is true only before the first step/action is taken
                    done = False # done is updated by termination rule
                    while not done:

                        pi_action = self.pid_controller(self.env.unwrapped.error)

                        ppo_action, next_hidden_state = self.ppo.predict(obs)
                        ppo_action = ppo_action.item()

                        action = pi_action + ppo_action

                        next_obs, reward, done, truncated, info = self.env.step(action)
                        steps_in_episode += 1

                        # Get value and log_prob
                        obs_tensor = pyTorch.tensor(obs, dtype=pyTorch.float32).unsqueeze(0)
                        action_tensor = pyTorch.tensor([ppo_action], dtype=pyTorch.float32)
                        with pyTorch.no_grad():
                            value = self.ppo.policy.predict_values(obs_tensor)
                            policy_generated_distribution = self.ppo.policy.get_distribution(obs_tensor)
                            log_prob = policy_generated_distribution.log_prob(action_tensor)


                        total_steps_counter += 1
                        self.ppo.rollout_buffer.add(obs, action, reward, is_start_episode, value, log_prob)
                        records.append((*obs, max(pi_action, 0), ppo_action, action, reward, self.env.unwrapped.error, steps_in_episode, is_start_episode))

                        obs = next_obs # Can update obs after storing in buffer
                        is_start_episode = False

                    # Post episode run
                    print(f"{steps_in_episode=}")
                    steps_in_episode = 0

            self.ppo.train()
            self.ppo.rollout_buffer.reset()
            self.ppo.logger.record("train/iteration", iteration)
            self.ppo.logger.dump(iteration)
        
        counter_divided_by_iterations = total_steps_counter/iterations
        is_same_as_buffer_size = counter_divided_by_iterations == self.buffer_size
        print(f"{self.buffer_size=}")
        print(f"{total_steps_counter=} divided by {iterations=} is equal to {counter_divided_by_iterations} ({is_same_as_buffer_size=})")

        
        pd.DataFrame(
            records, 
            columns=[f"x{i+1}" for i in range(self.env.unwrapped.x_size)] + \
                    ["y_ref", "z_t", "PID_action", "PPO_action", "action", "reward", "error", "steps_in_episode", "is_start_episode"]
        ).to_csv(
            f"{self.logs_folder_path}/records.csv", 
            index=False
        )
