from typing import Callable, Literal, Optional
import math
import re
import gymnasium
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import DiagGaussianDistribution
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
from save_file_utils import create_dir_if_not_exists
from modules.Scheduller import Scheduller



class CustomMLP_divided(nn.Module):
    def __init__(self, 
                 feature_dim: int,
                 last_layer_dim_pi: int = 1,
                 last_layer_dim_vf: int = 1,
                 neurons_per_layer: int = 6,
                 activation_function_name: Literal["no activation", "relu", "tanh", "swish"] = "no activation",
                 ):
        super().__init__()

        map_name_to_activation_function = {
            "no activation": lambda: None,
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            # "swish": nn.,
        }
        
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create the same network arquitecture for policy and value networks
        self.policy_upper_net, self.policy_lower_net, self.policy_merged_net = \
            self.create_divided_networks(feature_dim, last_layer_dim_pi, 
                                         neurons_per_layer, activation_function_name)
        self.value_upper_net, self.value_lower_net, self.value_merged_net = \
            self.create_divided_networks(feature_dim, last_layer_dim_vf, 
                                         neurons_per_layer, activation_function_name)
    
    def create_divided_networks(self, 
                        feature_dim: int, 
                        last_layer_dim: int,
                        neurons_per_layer: int,
                        activation_function: Literal["no activation", "relu", "tanh", "swish"],
                        ) -> tuple[nn.Sequential, nn.Sequential, nn.Sequential]:
        
        assert neurons_per_layer >= 6, "neurons_per_layers must be greater than 6."

        # Put 66% of neurons in upper net and rest 33% in lower net
        parts = neurons_per_layer // 6
        upper_net_neurons_per_layers = int(parts * 4)
        lower_net_neurons_per_layers = int(parts * 2)

        if activation_function == "no activation":
            upper_net = nn.Sequential(
                nn.Linear(feature_dim - 1, upper_net_neurons_per_layers),
                nn.Linear(upper_net_neurons_per_layers, upper_net_neurons_per_layers), # Finish with upper_net_neurons_per_layers
            )

            lower_net = nn.Sequential(
                nn.Linear(1, lower_net_neurons_per_layers),
                nn.Linear(lower_net_neurons_per_layers, lower_net_neurons_per_layers), # Finish with lower_net_neurons_per_layers
            )

            merged_net = nn.Sequential(
                nn.Linear(upper_net_neurons_per_layers + lower_net_neurons_per_layers, 
                          last_layer_dim), # Recives upper_net_neurons_per_layers + lower_net_neurons_per_layers from both previous finished layers
            )

        elif activation_function == "relu":
            upper_net = nn.Sequential(
                nn.Linear(feature_dim - 1, upper_net_neurons_per_layers),
                nn.ReLU(),
                nn.Linear(upper_net_neurons_per_layers, upper_net_neurons_per_layers), # Finish with upper_net_neurons_per_layers
                nn.ReLU()
            )

            lower_net = nn.Sequential(
                nn.Linear(1, lower_net_neurons_per_layers),
                nn.ReLU(),
                nn.Linear(lower_net_neurons_per_layers, lower_net_neurons_per_layers), # Finish with lower_net_neurons_per_layers
                nn.ReLU()
            )

            merged_net = nn.Sequential(
                nn.Linear(upper_net_neurons_per_layers + lower_net_neurons_per_layers, 
                          last_layer_dim), # Recives upper_net_neurons_per_layers + lower_net_neurons_per_layers from both previous finished layers
                nn.ReLU()
            )
        elif activation_function == "tanh":
            upper_net = nn.Sequential(
                nn.Linear(feature_dim - 1, upper_net_neurons_per_layers),
                nn.Tanh(),
                nn.Linear(upper_net_neurons_per_layers, upper_net_neurons_per_layers), # Finish with upper_net_neurons_per_layers
                nn.Tanh()
            )

            lower_net = nn.Sequential(
                nn.Linear(1, lower_net_neurons_per_layers),
                nn.Tanh(),
                nn.Linear(lower_net_neurons_per_layers, lower_net_neurons_per_layers), # Finish with lower_net_neurons_per_layers
                nn.Tanh()
            )

            merged_net = nn.Sequential(
                nn.Linear(upper_net_neurons_per_layers + lower_net_neurons_per_layers, 
                          last_layer_dim), # Recives upper_net_neurons_per_layers + lower_net_neurons_per_layers from both previous finished layers
                nn.Tanh()
            )

        return upper_net, lower_net, merged_net


    def forward(self, input_vector: pyTorch.Tensor):
        # # print(f"[CustomMLP forward] {input_vector=}")
        # print(f"log_std antes do treino: {self}")
        # print(f"log_std antes do treino: {dir(self)}")
        # print(f"log_std antes do treino: {self.ppo.policy.log_std}")

        # self.ppo.policy.log_std.data.clamp_(-2, 2)  # Mantém os valores dentro de uma faixa razoável

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
    

class CustomMLP(nn.Module):
    def __init__(self, 
                 feature_dim: int,
                 last_layer_dim_pi: int = 1,
                 last_layer_dim_vf: int = 1,
                 neurons_per_layer: int = 6,
                 activation_function_name: Literal["no activation", "relu", "tanh", "swish"] = "no activation"
                 ):
        super().__init__()

        map_choice_to_activation_function = {
            "no activation": lambda: None,
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            # "swish": nn.,
        }
        
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create the same network arquitecture for policy and value networks
        self.policy_net = self.create_single_network(feature_dim, last_layer_dim_pi, 
                                                     neurons_per_layer, activation_function_name)
        self.value_net = self.create_single_network(feature_dim, last_layer_dim_vf, 
                                                    neurons_per_layer, activation_function_name)

        
    def create_single_network(self, 
                              feature_dim: int, 
                              last_layer_dim: int,
                              neurons_per_layer: int,
                              activation_function: Literal["no activation", "relu", "tanh", "swish"],
                            ) -> nn.Sequential:

        if activation_function == "no activation":
            net = nn.Sequential(
                nn.Linear(feature_dim, neurons_per_layer),
                nn.Linear(neurons_per_layer, neurons_per_layer),
                nn.Linear(neurons_per_layer, last_layer_dim)
            )
        elif activation_function == "relu":
            net = nn.Sequential(
                nn.Linear(feature_dim, neurons_per_layer),
                nn.ReLU(),
                nn.Linear(neurons_per_layer, neurons_per_layer),
                nn.ReLU(),
                nn.Linear(neurons_per_layer, last_layer_dim),
                nn.ReLU()
            )
        elif activation_function == "tanh":
            net = nn.Sequential(
                nn.Linear(feature_dim, neurons_per_layer),
                nn.Tanh(),
                nn.Linear(neurons_per_layer, neurons_per_layer),
                nn.Tanh(),
                nn.Linear(neurons_per_layer, last_layer_dim),
                nn.Tanh()
            )
        
        return net


    def forward(self, input_vector: pyTorch.Tensor):
        # print(f"[CustomMLP forward] {input_vector=}")

        return self.forward_actor(input_vector), self.forward_critic(input_vector)


    def forward_actor(self, input_vector: pyTorch.Tensor) -> pyTorch.Tensor:
        # print(f"[CustomMLP forward_actor] {input_vector=}")
        return self.policy_net(input_vector)


    def forward_critic(self, input_vector: pyTorch.Tensor) -> pyTorch.Tensor:
        # print(f"[CustomMLP forward_critic] {input_vector=}")
        return self.value_net(input_vector)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, 
                 observation_space: gymnasium.spaces.Space,
                 action_space: gymnasium.spaces.Space,
                 lr_schedule: Callable[[float], float],
                 *args,
                 **kwargs,
                 ):
        kwargs["ortho_init"] = False # Disable orthogonal initialization

        self.__divide_neural_network = kwargs.pop("divide_neural_network")
        self.__neurons_per_layer = kwargs.pop("neurons_per_layer")
        self.__activation_function_name = kwargs.pop("activation_function_name")
        __adam_stepsize = kwargs.pop("adam_stepsize")
        __weight_decay = kwargs.pop("weight_decay")

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            optimizer_kwargs={
                "eps": 1e-5,
                # "lr": __adam_stepsize,
                "weight_decay": __weight_decay
            },
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        if self.__divide_neural_network:
            self.mlp_extractor = CustomMLP_divided(
                self.features_dim, 
                self.action_space.shape[0], 
                self.action_space.shape[0],
                neurons_per_layer=self.__neurons_per_layer,
                activation_function_name=self.__activation_function_name
            )
        else:
            self.mlp_extractor = CustomMLP(
                self.features_dim, 
                self.action_space.shape[0],
                self.action_space.shape[0],
                neurons_per_layer=self.__neurons_per_layer,
                activation_function_name=self.__activation_function_name
            )
    
    # def train(self, _: bool):
    #     self.optimizer.zero_grad()  # Zera os gradientes
    #     loss = self.compute_loss()  # Calcula a loss (exemplo)
    #     loss.backward()  # Propaga os gradientes
    #     self.optimizer.step()  # Atualiza os pesos





class PIME_PPO:
    """
    PIME = Prior_PI_controller & Integrator & Model Ensemble
    
    [en]
    An algorithm that uses PID and PPO.
    
    [pt-br]
    Um algoritmo que usa um controlador PID e o algoritmo de aprendizado por reforço PPO.
    """

    def __init__(self, 
                 env: gymnasium.Env,                                                   # gymnasium.Env like set point env
                 scheduller: Scheduller,                                               # manage the set_point at each step
                 ensemble_generator: EnsembleGenerator,                                # manage the random inicialization of parameters
                 ensemble_size: int = 2,                                               # number of models in ensemble
                 pid_controller: Optional[Callable[[float], float]] = None,            # Tuned pid encapsulated inside a function
                 Kp: np.float64 = 1,                                                   # 
                 Ki: np.float64 = 1,                                                   # 
                 Kd: np.float64 = 0,                                                   # Kd is not used in PIME PPO
                 tracked_point_name: str = 'x1',                                       # string like "x[integer]"
                 use_GPU: bool = False,                                                # False for CPU, True for GPU
                 logs_folder_path: str = "logs/ppo/",                                  # All logs and outputs will be saved here
                 buffer_size: Optional[int] = None,                                    # buffer_size = ensemble_size * episode_lenght (minimum size needed to not get buffer overflow error)
                 episodes_per_sample: int = 5,                                         # Number of episodes collected for one parameter set
                 ppo_verbose: Literal[0, 1, 2] = 0,                                    # PPO param - 0 1 2
                 discount = 0.99,                                                      # PPO param - Discount factor
                 clip_range: float = 0.2,                                              # PPO param
                 gae_lambda: float = 0.97,                                             # PPO param
                 vf_coef: float = 1.0,                                                 # PPO param c1 = vf_coef
                 ent_coef: float = 0.02,                                               # PPO param c2 = ent_coef
                 horizon: int = 200,                                                   # PPO param
                 adam_stepsize: float = 3e-4,                                          # PPO param (torch adam optimzer)
                 weight_decay: float = 1e-5,                                           # PPO param (torch adam optimzer)
                 minibatch_size: int = 256,                                            # PPO param
                 epochs: int = 10,                                                     # PPO param
                 divide_neural_network: bool = True,                                   # PPO neural net param
                 neurons_per_layer: int = 6,                                           # PPO neural net param
                 activation_function_name: Literal["no activation"] = "no activation", # PPO neural net param
                 integrator_bounds: tuple[int, int] = (-25, 25),                       # Clip para PID e PPO (formula do integrator)
                 agent_action_bounds: tuple[int, int] = (-1, 1),                       # Clip para PPO (foward que retorna resultado da rede neural)
                 pid_type: Literal["PID", "PI", "P"] = "PI",                           # PID, PI or P
                 sample_period: int = 1,                                               # Euler method sampling period (dt)
                 seed: Optional[int] = None,                                           # Seed for reproducibility
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
            self.pid_controller = PIDController(Kp, Ki, Kd, 
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
        
        self.device = pyTorch.device("cuda" if use_GPU else "cpu")
        

        """Lembrar:
            Definir buffer_size usando rollout_buffer_kwargs ou buffer_size não funciona. 
            Olhando o código fonte, precisa definir n_steps que é  "the number 
            of experiences which is collected from a single environment under 
            the current policy before its next update".
        """
        self.ppo = PPO(CustomActorCriticPolicy, 
                       env, 
                       device                = self.device,
                       verbose               = ppo_verbose,          #
                       gamma                 = discount,             # Discount factor
                       gae_lambda            = gae_lambda,           # Factor for trade-off of bias vs variance for Generalized Advantage Estimator. Equivalent to classic advantage when set to 1.
                       clip_range            = clip_range,           #
                       clip_range_vf         = None,                 # No clip applyed to vf
                       vf_coef               = vf_coef,              # c2=0.02
                       ent_coef              = ent_coef,             # c1=1.0
                       rollout_buffer_class  = RolloutBuffer,        #
                       seed                  = seed,                 #
                       # rollout_buffer_kwargs = {'buffer_size': self.buffer_size},
                       # buffer_size           = self.buffer_size,
                       # n_steps               = self.buffer_size,
                       n_steps               = horizon,              # Number of steps to run for each environment per update
                       learning_rate         = adam_stepsize,
                       batch_size            = minibatch_size,
                       n_epochs              = epochs,
                       policy_kwargs         = {
                            "divide_neural_network": divide_neural_network,
                            "neurons_per_layer": neurons_per_layer,
                            "activation_function_name": activation_function_name,
                            "adam_stepsize": adam_stepsize,
                            "weight_decay": weight_decay,
                            }
                       )
        
        # Setup logger (mandatory to avoid "AttributeError: 'PPO' object has no attribute '_logger'.")
        # https://stable-baselines3.readthedocs.io/en/master/common/logger.html
        create_dir_if_not_exists(logs_folder_path)
        new_logger = configure(logs_folder_path, ["stdout", "csv", "tensorboard"])
        self.ppo.set_logger(new_logger)


    def train(self, 
              steps_to_run = 100_000,
              extra_record_only_pid: bool = False,
              extra_record_only_agent: bool = False,
              should_save_records: bool = True,
              should_save_trained_model: bool = False,
            ) -> float:
        """
        steps_to_run [int]: Approximattely the ammount of calls to env.step function. The ammount of calls cannot be less than steps_to_run, but can be more if nedded to finish an iteration. Iteration is the ammount of times to loop through all models in ensemble. Each complete ensemble loop is 1 iteration.
        """

        assert steps_to_run > 0, "steps_to_run must be greater than 0."

        records = [] # (*x_vector, y_ref, z_t, PID_action, PPO_action, action, reward, error, steps_in_episode)
        
        # steps_extra_info: list[tuple] = []
        steps_in_episode = 0
        total_steps_counter = 0
        trainings = 0

        device = self.device

        steps_per_iteration = self.steps_per_episode * self.ensemble_size * self.episodes_per_sample
        iterations: int = math.ceil(steps_to_run / steps_per_iteration)
        # [uncomment] print(f"Steps requested: {steps_to_run}, Steps to be executed: {iterations * steps_per_iteration} (Iterations: {iterations})")

        def train_and_reset_ppo() -> None:
            self.ppo.train()
            self.ppo.rollout_buffer.reset()

        branchless_train_and_reset = {
            True: train_and_reset_ppo,
            False: lambda: None
        }

        for iteration in range(1, iterations+1):

            for m in range(self.ensemble_size):
                
                for j in range(self.episodes_per_sample):
                    sample_parameters = self.ensemble.generate_sample()
                    obs, truncated = self.env.reset(options = {"ensemble_sample_parameters": sample_parameters})
                    print(f"reset OBS @@@ {obs=} | {self.env.unwrapped.error=}")
                    done = False # done is updated by termination rule
                    steps_in_episode = 0
                    episode_reward = 0

                    while not done:

                        pi_action = self.pid_controller(self.env.unwrapped.error)
                
                        ppo_action, next_hidden_state = self.ppo.predict(obs)
                        ppo_action = 30.0 * ppo_action.item()

                        action = pi_action + ppo_action

                        # Other possible combinations
                        # action = pi_action + ppo_action * self.env.unwrapped.error
                        # action = pi_action * ppo_action

                        next_obs, reward, done, truncated, info = self.env.step(action)
                        # print(f"NEXT OBS @@@ {next_obs=} | {self.env.unwrapped.error=}")
                        # input(">>>")
                        steps_in_episode += 1
                        total_steps_counter += 1    
                        episode_reward += reward      

                        # Get value and log_prob
                        obs_tensor = pyTorch.tensor(obs, dtype=pyTorch.float32).unsqueeze(0).to(device)
                        action_tensor = pyTorch.tensor([ppo_action], dtype=pyTorch.float32).to(device)
                        with pyTorch.no_grad():
                            value = self.ppo.policy.predict_values(obs_tensor)
                            policy_distribution: DiagGaussianDistribution = self.ppo.policy.get_distribution(obs_tensor)
                            log_prob = policy_distribution.log_prob(action_tensor)


                        self.ppo.rollout_buffer.add(obs, action_tensor, reward, done, value, log_prob)
                        records.append((*next_obs, pi_action, ppo_action, action, reward, self.env.unwrapped.error, steps_in_episode))

                        obs = next_obs # Can update obs after storing in buffer
                        # if total_steps_counter % self.ppo.n_steps == 0:
                            
                        #     for name, param in self.ppo.policy.named_parameters():
                        #         if param.requires_grad:
                        #             print(f"{name}: max={param.data.max()}, min={param.data.min()}")
                        #         if param.grad is not None:
                        #             print(f"{name}: max_grad={param.grad.max()}, min_grad={param.grad.min()}")
                        #     print(f"log_std antes do treino: {self.ppo.policy.log_std}")
                        #     print(f"Ações armazenadas: {self.ppo.rollout_buffer.actions.mean()}, {self.ppo.rollout_buffer.actions.std()}, {self.ppo.rollout_buffer.actions.min()}, {self.ppo.rollout_buffer.actions.max()}")
                        #     print(f"--------------------------------- {trainings=}")
                        #     trainings += 1
                        # if trainings == 95:
                        #     pass
                        # (branchless) Treina e reseta o buffer ao atingir `n_steps`
                        branchless_train_and_reset[
                            total_steps_counter % self.ppo.n_steps == 0
                        ]()
                            
                    # End of while loop / end of episode run ###################

                    # [uncomment] print(f"last_steps_in_episode={steps_in_episode}")
                    # [uncomment] last_steps_in_episode = steps_in_episode
                    last_episode_reward = episode_reward # Show how well the trained model is performing

            # self.ppo.train()
            # self.ppo.rollout_buffer.reset()
            self.ppo.logger.record("train/iteration", iteration)


        self.ppo.logger.dump(total_steps_counter)
        
        # Print last ppo logs
        # msg = "Última iteração de treinamento: "
        # print("+" + "-" * len(msg) + "+")
        # print(f"|{msg}" + "|")
        # for key, value in last_log[0].items():
        #     print(f"| {key}: {value}" + " " * (len(msg) - len(key) - len(str(value)) - 3) + "|")
        # print("+" + "-" * len(msg) + "+")

        # [uncomment] counter_divided_by_iterations = total_steps_counter/iterations
        # [uncomment] is_same_as_buffer_size = counter_divided_by_iterations == self.buffer_size
        # [uncomment] print(f"{self.buffer_size=}")
        # [uncomment] print(f"{total_steps_counter=} divided by {iterations=} is equal to {counter_divided_by_iterations} ({is_same_as_buffer_size=})")

        if (should_save_records):
            pd.DataFrame(
                records, 
                columns=[f"x{i+1}" for i in range(self.env.unwrapped.x_size)] + \
                        ["y_ref", "z_t", "PID_action", "PPO_action", "combined_action", "reward", "error", "steps_in_episode"]
            ).to_csv(
                f"{self.logs_folder_path}/records.csv", 
                index=False
            )

            # save last used sample_parameters
            pd.DataFrame(
                sample_parameters.items(), 
                columns=["parameter_name", "parameter_value"]
            ).to_json(
                f"{self.logs_folder_path}/last_sample_parameters.json", 
                orient="records"
            )

            

        if (should_save_trained_model):
            self.ppo.save(f"{self.logs_folder_path}/trained_ppo_model")

            policy_weights = self.ppo.policy.state_dict()
            # pyTorch.save(policy_weights, f"{self.logs_folder_path}/trained_ppo_policy_weights.pth")
            with open(f"{self.logs_folder_path}/trained_ppo_policy_weights.txt", "w") as f:
                f.write(str(policy_weights)) # Save in plain text
        
        if (extra_record_only_pid and should_save_records):
            self.record_pid_episode(sample_parameters)

        if (extra_record_only_agent and should_save_records):
            self.record_ppo_episode(sample_parameters)

        return last_episode_reward


    def record_pid_episode(self, sample_parameters: dict[str, float | int | np.ndarray], logs_folder_path: Optional[str] = None) -> None:
        """
        Records the PID controller response in a single episode and saves it in a csv file.
        """
        if logs_folder_path is not None:
            create_dir_if_not_exists(logs_folder_path)
        records = []
        done = False
        steps_in_episode = 0
        obs, truncated = self.env.reset(options = {"ensemble_sample_parameters": sample_parameters})
        # execute 1 episode
        while not done:
            pi_action = self.pid_controller(self.env.unwrapped.error)
            next_obs, reward, done, truncated, info = self.env.step(pi_action) 
            steps_in_episode += 1
            records.append((*obs, pi_action, reward, self.env.unwrapped.error, steps_in_episode))

            obs = next_obs # Can update obs after storing in buffer
        
        pd.DataFrame(
            records, 
            columns=[f"x{i+1}" for i in range(self.env.unwrapped.x_size)] + \
                    ["y_ref", "z_t", "PID_action", "reward", "error", "steps_in_episode"]
        ).to_csv(
            f"{self.logs_folder_path}/only_pid_records.csv", 
            index=False
        )


    def record_ppo_episode(self, 
                           sample_parameters: dict[str, float | int | np.ndarray], 
                           policy_weights: Optional[dict[str, any]] = None
                           ) -> None:
        """
        Records the PPO without PID controller response in a single episode and saves it in a csv file.
        """
        records = []
        done = False
        steps_in_episode = 0
        obs, truncated = self.env.reset(options = {"ensemble_sample_parameters": sample_parameters})

        # override ppo weights
        if policy_weights is not None:
            self.ppo.policy.load_state_dict(policy_weights)

        # Execute one episode
        while not done:
            ppo_action, next_hidden_state = self.ppo.predict(obs)
            ppo_action = ppo_action.item()
            next_obs, reward, done, truncated, info = self.env.step(ppo_action) 
            steps_in_episode += 1
            records.append((*obs, ppo_action, reward, self.env.unwrapped.error, steps_in_episode))

            obs = next_obs
        
        pd.DataFrame(
            records, 
            columns=[f"x{i+1}" for i in range(self.env.unwrapped.x_size)] + \
                    ["y_ref", "z_t", "PPO_action", "reward", "error", "steps_in_episode"]
        ).to_csv(
            f"{self.logs_folder_path}/only_ppo_records.csv", 
            index=False
        )
        

        