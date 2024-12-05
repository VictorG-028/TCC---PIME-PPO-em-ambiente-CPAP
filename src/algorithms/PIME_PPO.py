from typing import Callable, Optional
import math
import os
import gymnasium
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.logger import configure

import torch as pyTorch
import torch.nn.functional as torch_nn_functional
import torch.nn as nn

from enums.TerminationRule import TerminationRule
from enums.ErrorFormula import ErrorFormula
from modules.Ensemble import Ensemble
from modules.Scheduller import Scheduller

Y_REF = 2

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


class PIDController:
    """
    k = pi controller
    y_t = current y value
    error = error_formula(y, y_ref) # Ex.: (y_ref - y); -(y_ref - y)²
    z_0 = 0
    z_t = z_(t-1) + error
    k(y_t, y_ref) = -Kp * error +Ki * z_t
    
    [en]
    PID controller for a discrete-time state space model (this means that dt = 1).
    Consider that env.step() takes roughly the same time and env uses a discrete-time model.
    This pid controller is meant to recive optimized kp, ki and kd.
    
    [pt-br]
    Controlador PID para um modelo de espaço de estados de tempo discreto (isso significa que dt = 1).
    dt = 1 por que env.step() demora + ou - o mesmo tempo e o env usa um modelo de tempo discreto.
    Este controlador pid espera receber kp, ki e kd otimizados.
    """
    def __init__(
            self, 
            Kp, Ki, Kd, 
            integrator_bounds: list[int, int],
            dt = 1,
            error_formula: ErrorFormula | Callable[[float, float], float] = ErrorFormula.DIFFERENCE, 
            use_derivative: bool = False
            ):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.min = integrator_bounds[0]
        self.max = integrator_bounds[1]
        self.dt = dt
        self.error_formula = error_formula
        self.previous_error = 0
        self.integral = 0

        # Pergunta: Porque os exeprimentos do tanque de água e controle de ph usam dt = 2 e dt = 20 ? 
        # Pergunta: Qual é o dt usado no CPAP ?
        # Pergunta: Porque dt é usado na formula do PID ?
        # Pergunta: O que é discretização do método de Euler ?

        def PID_formula(error):
            derivative = (error - self.previous_error) / self.dt
            self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        def PI_formula(error):
            return self.Kp * error + self.Ki * self.integral
        
        self.formula = PID_formula if use_derivative else PI_formula

    def __call__(self, error: float) -> float:
        self.integral += np.clip(error * self.dt, self.min, self.max)
        output = self.formula(error)
        self.previous_error = error

        return output



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
                 ensemble: Ensemble,                                        # manage the random inicialization of parameters
                 pid_controller: Optional[Callable[[float], float]] = None, # Tuned pid encapsulated inside a function
                 optimized_Kp: np.float64 = 1,                              # 
                 optimized_Ki: np.float64 = 1,                              # 
                 optimized_Kd: np.float64 = 0,                              # Kd is not used in PIME PPO
                 tracked_point_name: str = 'x1',                            # string like "x[integer]"
                 verbose: int = 1,                                          # 
                 logs_folder_path: str = "logs/ppo/",                       # PPO logs
                 buffer_size: Optional[int] = None,                         # buffer_size = ensemble.size * episode_lenght (minimum size needed to not get buffer overflow error) # TODO: perceber q largura do episódio depende de teremination rule e dar um jeito de descobrir quanto vai durar
                 episodes_per_sample: int = 5,                              # Number of episodes collected for one parameter set
                 gae_lambda: float = 0.97,                                  # PPO param
                 c1: float = 1.0,                                           # not used
                 c2: float = 0.02,                                          # PPO param
                 integrator_bounds: list[int, int] = [-25, 25],             # Clip para pid (formula do integrator)
                 sample_period: int = 1                                     # Euler method sampling period (dt)
                 ) -> None:
        
        assert env.unwrapped.error_formula is not None, \
            "Must have an error_formula (type ErrorFormula or Callable[y:float, y_ref:float], output:float]) propertie inside env."
        
        if pid_controller is not None:
            self.pid_controller = pid_controller
        else:
            self.pid_controller = PIDController(optimized_Kp, optimized_Ki, optimized_Kd, integrator_bounds, sample_period, env.unwrapped.error_formula) 

        self.env = env
        self.scheduller = scheduller
        self.ensemble = ensemble
        self.tracked_point_name = tracked_point_name
        self.episodes_per_sample = episodes_per_sample
        self.sample_period = sample_period

        # flat_obs_space = flatten_observation_space(self.env.observation_space)
        # print(buffer_size)
        # print(self.env.observation_space)
        # print(self.env.action_space)
        # print(flat_obs_space, type(flat_obs_space))
        # print("@@@@@@@@@@@@@@@@@@")

        # TODO: esse if engessa o código da classe TerminationRule, é preciso encotrar uma outra forma de obter steps_per_episode (talvez colocar o input direto no __init__, mas isso permite o step do BaseEnv calcular self.done de forma que gere o erro de falta de memória no buffer ou gere uma situação sem erros, mas que usa mais emmória no buffer do que o necessário. Colocar steps_per_episode como input no __init__ do PIME_PPO.py também deixa o código difícil de usar e o ideal eh não criar dificuldade)
        if env.unwrapped.max_step is not None:
            self.steps_per_episode = env.unwrapped.max_step
        else:
            self.steps_per_episode = env.unwrapped.scheduller.intervals_sum

        if buffer_size is None:
            buffer_size = self.steps_per_episode * ensemble.size * episodes_per_sample # * self.scheduller.intervals_sum 
                          
        self.buffer_size = buffer_size
            
        print(f"{buffer_size=}")
        self.ppo = PPO(CustomActorCriticPolicy, 
                       env, 
                       verbose               = verbose,              #
                       vf_coef               = c2,                   #
                       gae_lambda            = gae_lambda,           # Factor for trade-off of bias vs variance for Generalized Advantage Estimator. Equivalent to classic advantage when set to 1.
                       gamma                 = 0.99,                 # Discount factor
                       rollout_buffer_class  = RolloutBuffer,        #
                       # Definir o buffer_size usando rollout_buffer_kwargs ou buffer_size não funciona. Olhando o código fonte, precisa definir n_steps que é "the number of experiences which is collected from a single environment under the current policy before its next update"
                    #    rollout_buffer_kwargs = {'buffer_size': buffer_size},
                    #    buffer_size           = buffer_size,
                       n_steps               = self.buffer_size # TODO: descobrir por que remover shceduller.intervals_sum calcula corretamente o n_envs (lembrar: n_envs = buffer size, uma vez que num_envs = 1)
                       )
        
        # Setup logger (mandatory to avoid "AttributeError: 'PPO' object has no attribute '_logger'.")
        # https://stable-baselines3.readthedocs.io/en/master/common/logger.html
        if not os.path.exists(logs_folder_path):
            os.makedirs(logs_folder_path)
            print(f"[Aviso] Pasta '{logs_folder_path}' foi criada para armazenar os logs.")
        new_logger = configure(logs_folder_path, ["stdout", "csv", "tensorboard"])
        self.ppo.set_logger(new_logger)

        # TODO: usar REGEX na string tracked_point_name para validar que é "x[:number:]"


    def train(self, steps_to_run = 100_000) -> None:
        """
        steps_to_run [int]: Approximattely the ammount of calls to env.step function. The ammount of calls cannot be less than steps_to_run, but can be more if nedded to finish an iteration. Iteration is the ammount of times to loop through all models in ensemble. Each complete ensemble loop is 1 iteration.
        """

        assert steps_to_run > 0, "steps_to_run must be greater than 0."
        
        # steps_extra_info: list[tuple] = []
        steps_in_episode = 0
        total_steps_counter = 0

        steps_per_iteration = self.steps_per_episode * self.ensemble.size * self.episodes_per_sample
        iterations: int = math.ceil(steps_to_run / steps_per_iteration)
        print(f"Steps requested: {steps_to_run}, Steps to be executed: {iterations * steps_per_iteration} (Iterations: {iterations})")

        for iteration in range(1, iterations+1):

            for sample_parameters in self.ensemble.get_all_samples():
                self.env.unwrapped.set_enviroment_params(sample_parameters)
                
                for j in range(self.episodes_per_sample):
                    obs, truncated = self.env.reset()
                    is_start_episode = True # Is true only before the first step/action is taken
                    done = False # done is updated by termination rule
                    while not done:

                        pi_action = self.pid_controller(self.env.unwrapped.reward)

                        ppo_action, next_hidden_state = self.ppo.predict(obs)

                        action = pi_action + ppo_action.item()

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
