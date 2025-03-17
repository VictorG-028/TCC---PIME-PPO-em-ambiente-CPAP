from collections import namedtuple
from typing import Callable, Literal, Optional
import math
import re
import gymnasium
import numpy as np
import pandas as pd
import copy

import torch as pyTorch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from algorithms.PID_Controller import Batch_PIDController, PIDController
from enums.TerminationRule import TerminationRule
from enums.ErrorFormula import ErrorFormula
from modules.EnsembleGenerator import EnsembleGenerator
from save_file_utils import create_dir_if_not_exists
from modules.Scheduller import Scheduller


class ANN_divided(nn.Module):
    def __init__(self, 
                obs_size, 
                hidden_sizes, 
                out_size,
                out_multiplier=1.0, 
                pid=lambda x: 0, 
                env_name = "double_water_tank",
                activation_function: Literal["no activation", "relu", "tanh"] = "no activation",
                use_activation_func_in_last_layer: bool = False
                ):
        super(ANN_divided, self).__init__()
        # self.set_biases_to_zero()

        out_multiplier = 30.0
        self.multiplier = pyTorch.tensor(out_multiplier, dtype=pyTorch.float32)

        self.pid = pid
        # Batch_PIDController(kp, ki, kd, 
        #                     integrator_bounds=(-25, 25),
        #                     dt=2,
        #                     controller_type="PI",
        #                     )

        map_name_to_activation_function = {
            "no activation": lambda: None,
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
        }
        self.activation_function = map_name_to_activation_function.get(activation_function, nn.Identity)

        if env_name == "double_water_tank":
            def error_extractor(input_vector: pyTorch.Tensor):
                    # x1 = input_vector[:, 0].unsqueeze(1)
                    x2 = input_vector[:, 1].unsqueeze(1)
                    # x3 = input_vector[:, 2].unsqueeze(1) # last action (u_t)
                    yRef = input_vector[:, 3].unsqueeze(1)
                    # z_t = input_vector[:, 4].unsqueeze(1)
                    return yRef - x2 
            self.error_extractor = error_extractor
        elif env_name == "CPAP":
            def error_extractor(input_vector: pyTorch.Tensor):
                    # x1 = input_vector[:, 0].unsqueeze(1)
                    # x2 = input_vector[:, 1].unsqueeze(1)
                    x3 = input_vector[:, 2].unsqueeze(1)
                    yRef = input_vector[:, 3].unsqueeze(1)
                    # z_t = input_vector[:, 4].unsqueeze(1)
                    # u_last = input_vector[:, 5].unsqueeze(1)
                    # p_last = input_vector[:, 6].unsqueeze(1)
                    # one_hot_1 = input_vector[:, 7].unsqueeze(1)
                    # one_hot_2 = input_vector[:, 8].unsqueeze(1)
                    # one_hot_3 = input_vector[:, 9].unsqueeze(1)
                    return yRef - x3
            self.error_extractor = error_extractor

        self.split_input = lambda input_vector: (input_vector[:, :-1], input_vector[:, -1].unsqueeze(1))

        # Definição das camadas separadas
        assert hidden_sizes[0] >= 6, "The first hidden layer must have at least 6 neurons."
        neurons_per_layer = hidden_sizes[0]
        parts = neurons_per_layer // 6
        upper_net_neurons_per_layer = int(parts * 4)  # 4 partes para a parte superior
        lower_net_neurons_per_layer = int(parts * 2)  # 2 partes para a parte inferior

        # Rede superior (x_values_and_yRef_t)
        self.policy_upper_net = self._build_mlp(obs_size - 1, upper_net_neurons_per_layer, activation_function)

        # Rede inferior (z_t)
        self.policy_lower_net = self._build_mlp(1, lower_net_neurons_per_layer, activation_function)

        # Rede de fusão
        merge_size = upper_net_neurons_per_layer + lower_net_neurons_per_layer
        if use_activation_func_in_last_layer and activation_function != "no activation":
            merged_net = nn.Sequential(
                nn.Linear(merge_size, out_size),
                self.activation_function()
            )
            raise NotImplementedError("Não deveria entrar aqui.")
        else:
            merged_net = nn.Sequential(
                nn.Linear(merge_size, out_size)
            )
        self.policy_merged_net = merged_net

    def _build_mlp(self, input_size, neurons_per_layer, activation_function):
        """Cria um MLP de 3 camadas com ativação opcional"""
        layers = []
        prev_size = input_size

        for _ in range(3):  # Sempre 3 camadas
            layers.append(nn.Linear(prev_size, neurons_per_layer))
            if activation_function != "no activation":
                layers.append(self.activation_function())
            prev_size = neurons_per_layer

        return nn.Sequential(*layers)

    def forward(self, input_vector: pyTorch.Tensor):
        x_values_and_yRef_t, z_t = self.split_input(input_vector)

        upper_tensor = self.policy_upper_net(x_values_and_yRef_t)
        lower_tensor = self.policy_lower_net(z_t)
        merged_tensor = pyTorch.cat((upper_tensor, lower_tensor), dim=1)
        out = self.policy_merged_net(merged_tensor)

        error = self.error_extractor(input_vector)
        # pid_action = self.pid.forward(error)
        # out = out * self.multiplier 
        # out = out + pid_action
        return out * self.multiplier # + self.pid.forward(error)
    
    def reset_pid(self):
        self.pid.reset()

    def set_biases_to_zero(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.bias)  # Define todos os bias como zero


class ANN(nn.Module):
    def __init__(self, 
                obs_size, 
                hidden_sizes, 
                out_size,
                out_multiplier=1.0, 
                # recorder = lambda out: None,
                pid: Batch_PIDController = lambda input_tensor: 0,   
                env_name = "double_water_tank", # TODO: remove env_name parameter
                activation_function: Literal["no activation", "relu", "tanh"] = "no activation",
                use_activation_func_in_last_layer: bool = False
                ):
        super(ANN, self).__init__()
        self.multiplier = pyTorch.tensor(out_multiplier, dtype=pyTorch.float32)
        prev_size = obs_size
        layers = []

        self.pid = pid 
        # self.recorder = recorder
        # Batch_PIDController(kp, ki, kd, 
                                    #    integrator_bounds=(-25, 25),
                                    #    dt=2,
                                    #    controller_type="PI",
                                    #    )

        map_name_to_activation_function = {
            "no activation": lambda: None,
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
        }
        self.activation_function = map_name_to_activation_function.get(activation_function, None)
        assert self.activation_function is not None, f"Activation function {activation_function} must be Literal['no activation', 'relu', 'tanh']."

        # TODO: esses ifs pode ser removido ao colocar o tracked point sempre no "x1" e yRef sempre na segunda coluna
        if env_name == "double_water_tank":
            def error_extractor(input_vector: pyTorch.Tensor) -> pyTorch.Tensor:
                # x1 = input_vector[:, 0].unsqueeze(1)  # water height tank 1
                x2 = input_vector[:, 1].unsqueeze(1)    # water height tank 2
                yRef = input_vector[:, 2].unsqueeze(1)  # water height tank 2 reference
                # z_t = input_vector[:, 3].unsqueeze(1) # error acumm
                return yRef - x2
            self.error_extractor = error_extractor
        elif env_name == "CPAP":
            def error_extractor(input_vector: pyTorch.Tensor):
                # x1 = input_vector[:, 0].unsqueeze(1)  # flow
                # x2 = input_vector[:, 1].unsqueeze(1)  # volume
                x3 = input_vector[:, 2].unsqueeze(1)    # pressure
                yRef = input_vector[:, 3].unsqueeze(1)  # pressure
                # z_t = input_vector[:, 4].unsqueeze(1) # error acumm
                # u_last = input_vector[:, 5].unsqueeze(1) # last action
                # p_last = input_vector[:, 6].unsqueeze(1) # last pressure 
                # one_hot_1 = input_vector[:, 7].unsqueeze(1) # is exhaling
                # one_hot_2 = input_vector[:, 8].unsqueeze(1) # is inhaling
                # one_hot_3 = input_vector[:, 9].unsqueeze(1) # is pausing
                return yRef - x3
            self.error_extractor = error_extractor

        self.split_input = lambda input_vector: (input_vector[:, :-1], input_vector[:, -1].unsqueeze(1))

        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            if activation_function != "no activation":
                layers.append(self.activation_function())
            prev_size = size

        layers.append(nn.Linear(prev_size, out_size))
        if use_activation_func_in_last_layer and activation_function != "no activation":
            layers.append(self.activation_function())

        self.net = nn.Sequential(*layers)

    def forward(self, input_vector: pyTorch.Tensor):
        out: pyTorch.Tensor = self.net(input_vector)

        # self.recorder(out.tolist())

        # error = self.error_extractor(input_vector)

        return out * self.multiplier # + self.pid.forward(error)

    def reset_pid(self):
        self.pid.reset()
  


class TD3_Critic(nn.Module):
  def __init__(self, state_size, num_actions, hidden_sizes, 
               out_multiplier: int,
               is_divided: bool, 
               pid: Batch_PIDController, 
               env_name: Literal["double_water_tank", "CPAP"],
               activation_function: Literal["no activation", "relu", "tanh"],
               use_activation_func_in_last_layer: bool
               ):
    super(TD3_Critic, self).__init__()
    if is_divided:
        self.q1_net = ANN_divided(state_size + num_actions, hidden_sizes, 1, out_multiplier, pid, env_name, activation_function, use_activation_func_in_last_layer)
        self.q2_net = ANN_divided(state_size + num_actions, hidden_sizes, 1, out_multiplier, pid, env_name, activation_function, use_activation_func_in_last_layer)
    else:
        self.q1_net = ANN(state_size + num_actions, hidden_sizes, 1, out_multiplier, pid, env_name, activation_function, use_activation_func_in_last_layer)
        self.q2_net = ANN(state_size + num_actions, hidden_sizes, 1, out_multiplier, pid, env_name, activation_function, use_activation_func_in_last_layer)

  def forward(self, obs, act):
    t = pyTorch.cat((obs, act), dim=1)
    q1_out = self.q1_net(t)
    q2_out = self.q2_net(t)
    return q1_out, q2_out
  
  def forward_q1(self, obs, act):
    t = pyTorch.cat((obs, act), dim=1)
    return self.q1_net(t)
  
  def reset_pid(self):
    self.q1_net.reset_pid()
    self.q2_net.reset_pid()


class SyncNet:
  def __init__(self, model):
    self.sync_model = copy.deepcopy(model)

  @pyTorch.no_grad()
  def __call__(self, *args):
    # calculates the network output, without calculating the gradients
    return self.sync_model(*args)

  def update(self, model, decay):
    """
    Blend params of target net with params from the model
    :param decay:
    """
    assert isinstance(decay, float)
    assert 0.0 <= decay <= 1.0
    for sync_param, param in zip(self.sync_model.parameters(), model.parameters()):
      assert (sync_param.data.shape == param.data.shape)
      sync_param.data.copy_(sync_param.data*decay + param.data*(1.0 - decay))


# class ANN_Recorder:
#     action_buffer: list[float]
#     noise_buffer: list[float]

#     def __init__(self):
#         self.action_buffer = []
#         self.noise_buffer = []

#     def add_action(self, action: float):
#         self.action_buffer.append(action)

#     def add_noise(self, noise: float):
#         self.noise_buffer.append(noise)

#     def reset(self):
#         self.action_buffer = []
#         self.noise_buffer = []

#     def __call__(self, action: float):
#         self.action_buffer.append(action)
    
#     def __len__(self):
#         return len(self.action_buffer)

class TD3_Agent:
    def __init__(self, 
                state_size, 
                num_actions, 
                action_bounds: tuple[float, float], 
                mu_lr, q_lr, 
                target_decay, 
                hidden_sizes=[400, 300], 
                pid=lambda x: 0, 
                env_name="double_water_tank",
                is_divided: bool = False,
                activation_function: Literal["no activation", "relu", "tanh"] = "no activation",
                use_activation_func_in_last_layer: bool = False,
                device: Literal["cpu", "cuda"] = "cpu"
                ):

        self.device = device # pyTorch.device(device)
        self.min_action_bound = action_bounds[0]
        self.max_action_bound = action_bounds[1]
        # self.ann_recorder = ANN_Recorder()

        if is_divided:
            self.mu_net      = ANN_divided(state_size, hidden_sizes, num_actions, 
                                    out_multiplier=self.max_action_bound, 
                                    pid=pid, 
                                    env_name=env_name, 
                                    activation_function=activation_function, 
                                    use_activation_func_in_last_layer=use_activation_func_in_last_layer
                                    # recorder=lambda out: self.ann_recorder(out)
                                )
        else:
            self.mu_net       = ANN(state_size, hidden_sizes, num_actions, 
                                    out_multiplier=self.max_action_bound, 
                                    pid=pid, 
                                    env_name=env_name, 
                                    activation_function=activation_function, 
                                    use_activation_func_in_last_layer=use_activation_func_in_last_layer
                                    # recorder=lambda out: self.ann_recorder(out)
                                )
        
        self.q_net       = TD3_Critic(state_size, 
                                      num_actions, 
                                      hidden_sizes, 
                                      self.max_action_bound, 
                                      is_divided, 
                                      pid, 
                                      env_name, 
                                      activation_function, 
                                      use_activation_func_in_last_layer
                                    )
        self.mu_targ_net = SyncNet(self.mu_net)
        self.q_targ_net  = SyncNet(self.q_net)
        print(self.mu_net)
        print(self.q_net)
        self.num_actions = num_actions
        

        # Train each network separately
        self.mu_optimizer = optim.Adam(
            filter(lambda p: p.requires_grad and len(p.shape) > 1, self.mu_net.parameters()), 
            lr=mu_lr
        )
        self.q_optimizer = optim.Adam(
            filter(lambda p: p.requires_grad and len(p.shape) > 1, self.q_net.parameters()),
            lr=q_lr
        )
        self.tgt_decay = target_decay   # soft update factor used in the target networks; it is the complement of 'tau' (1.0 - tau)

    def reset_pid(self):
        self.q_net.reset_pid()
        self.mu_net.reset_pid()
        self.mu_targ_net.sync_model.reset_pid()
        self.mu_targ_net.sync_model.reset_pid()

        # self.ann_recorder.reset()

    def save(self, dir_path: str):
        pyTorch.save(self.mu_net.state_dict(), f'{dir_path}/actor.pth')
        pyTorch.save(self.q_net.state_dict(), f'{dir_path}/critic.pth')

    def load(self, dir_path: str, state_dict: Optional[dict] = None):
        if (state_dict):
            self.mu_net.load_state_dict(state_dict['actor'])
            self.q_net.load_state_dict(state_dict['critic'])
        else:
            self.mu_net.load_state_dict(pyTorch.load(f'{dir_path}/actor.pth'))
            self.q_net.load_state_dict(pyTorch.load(f'{dir_path}/critic.pth'))

    def select_action(self, state, noise=0.1, noise_clip=None):
        """ Select an appropriate action from the agent policy
            Args:
            state (array): current state of environment
            noise (float): how much noise to add to actions
            Returns:
            actions (float array): actions clipped within action range
        """
        state = pyTorch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action: pyTorch.Tensor = self.mu_net(state)
        action = action.squeeze(dim=0).detach().numpy() 
        if noise != 0: 
            noise = np.random.normal(0, noise, size=self.num_actions)
        if noise_clip is not None:
            noise = noise.clip(-noise_clip, noise_clip)
        # self.ann_recorder.add_noise(noise)
        action += noise
        return action # .clip(self.min_action_bound, self.max_action_bound)

    def update(self, replay_buffer, iterations, batch_size=100, discount=0.99, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        """ Train and update actor and critic networks
            Args:
            replay_buffer (ReplayBuffer): buffer for experience replay
            iterations (int): how many times to run training
            batch_size(int): batch size to sample from replay buffer
            discount (float): discount factor
            ...
            Return:
            critic_loss (float): loss from critic network
            actor_loss (float): loss from actor network
        """
        q_losses = []
        mu_losses = []
        for iteration in range(iterations):
            # Sample replay buffer 
            batch = replay_buffer.sample(batch_size)
            s1 = pyTorch.FloatTensor(batch.obs).to(self.device)
            a1 = pyTorch.FloatTensor(batch.actions).to(self.device)
            s2 = pyTorch.FloatTensor(batch.next_obs).to(self.device)
            not_done = pyTorch.FloatTensor(1 - batch.dones).unsqueeze(-1).to(self.device)
            reward = pyTorch.FloatTensor(batch.rews).unsqueeze(-1).to(self.device)

            # Select action according to policy and add clipped noise 
            # it is not the same as calling select_action, because: 
            # (1) in that function, the actions outputed are detached from the computation graph
            # (2) here, a batch of states is processed, so the shape is different
            noise = np.random.normal(0, policy_noise, size=a1.shape)
            noise = noise.clip(-noise_clip, noise_clip)
            noise = pyTorch.as_tensor(noise, dtype=pyTorch.float32)
            a2 = self.mu_targ_net(s2) + noise
            a2.clamp_(self.min_action_bound, self.max_action_bound)

            # Compute the target Q value
            target_q1, target_q2 = self.q_targ_net(s2, a2)
            target_q = pyTorch.min(target_q1, target_q2)
            target_q = reward + (not_done * discount * target_q)
            target_q.detach_() # it probably works without this

            # Get current Q estimates
            q1_s1_a1, q2_s1_a1 = self.q_net(s1, a1)

            # Compute critic loss
            q_loss = F.mse_loss(q1_s1_a1, target_q) + F.mse_loss(q2_s1_a1, target_q)
            q_losses.append(q_loss.item())

            # Optimize the critic
            self.q_optimizer.zero_grad()
            q_loss.backward()
            self.q_optimizer.step()

            # Delayed policy updates
            if iteration % policy_freq == 0:
                # Compute actor loss
                mu_loss = -self.q_net.forward_q1(s1, self.mu_net(s1))
                mu_loss = mu_loss.mean()
                mu_losses.append(mu_loss.item())

                # Optimize the actor 
                self.mu_optimizer.zero_grad()
                mu_loss.backward()
                self.mu_optimizer.step()

                # Target networks update
                self.q_targ_net.update(self.q_net, self.tgt_decay)
                self.mu_targ_net.update(self.mu_net, self.tgt_decay)
        
        return q_losses, mu_losses



Batch = namedtuple('Batch',['obs','actions','rews', 'next_obs', 'dones']) 
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.max_size = size
        self._init_buffers(obs_dim, act_dim, size)

    def _init_buffers(self, obs_dim, act_dim, size):
        """Inicializa ou reseta os buffers do replay buffer."""
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size = 0, 0  # Resetando ponteiros

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return Batch(obs=self.obs_buf[idxs],
                     actions=self.acts_buf[idxs],
                     next_obs=self.next_obs_buf[idxs],
                     rews=self.rews_buf[idxs],
                     dones=self.done_buf[idxs])

    def reset(self):
        """Reseta o buffer, limpando todas as experiências armazenadas."""
        self._init_buffers(self.obs_buf.shape[1], self.acts_buf.shape[1], self.max_size)

  

class PIME_TD3:
    """
    PIME = Prior_PI_controller & Integrator & Model Ensemble
    
    [en]
    An algorithm that uses PID and TD3 (variation of DDPG).
    
    [pt-br]
    Um algoritmo que usa um controlador PID e o algoritmo de aprendizado por reforço TD3 (variação de DDPG).
    """

    def __init__(self, 
                 env: gymnasium.Env,                                                   # gymnasium.Env like set point env
                 scheduller: Scheduller,                                               # manage the set_point at each step
                 ensemble_generator: EnsembleGenerator,                                # manage the random inicialization of parameters
                 ensemble_size: int = 2,                                               # number of models in ensemble
                 pid_controller: Optional[Callable[[float], float]] = None,            # Tuned pid encapsulated inside a function
                 Kp: np.float64 = 1,                                                   # 
                 Ki: np.float64 = 1,                                                   # 
                 Kd: np.float64 = 0,                                                   # Kd is not used in PIME TD3_DDPG
                 env_name: str = 'double_water_tank',                                  # string like "x[integer]"
                 tracked_point_name: str = 'x1',                                       # string like "x[integer]"
                 use_GPU: bool = False,                                                # False for CPU, True for GPU
                 logs_folder_path: str = "logs/ppo/",                                  # All logs and outputs will be saved here
                 buffer_size: Optional[int] = None,                                    # buffer_size = ensemble_size * episode_lenght (minimum size needed to not get buffer overflow error)
                 episodes_per_sample: int = 5,                                         # Number of episodes collected for one parameter set
                 discount = 0.99,                                                      # TD3_DDPG param - Discount factor
                 target_decay: float = 0.97,                                           # TD3_DDPG param
                 mu_lr: float = 3e-4,                                                  # TD3_DDPG param
                 q_lr: float = 3e-4,                                                   # TD3_DDPG param
                 batch_size: int = 256,                                                # TD3_DDPG param
                 epochs: int = 10,                                                     # TD3_DDPG param
                 policy_update_freq: int = 4,                                          # TD3_DDPG param
                 noise: float = 0.1,                                                   # TD3_DDPG param
                 noise_clip: float = 0.5,                                              # TD3_DDPG param
                 divide_neural_network: bool = True,                                   # TD3_DDPG neural net param
                 neurons_per_layer: int = 6,                                           # TD3_DDPG neural net param
                 activation_function_name: Literal["no activation"] = "no activation", # TD3_DDPG neural net param
                 use_activation_func_in_last_layer: bool = False,                      # TD3_DDPG neural net param
                 integrator_bounds: tuple[int, int] = (-25, 25),                       # Clip para PID e TD3_DDPG (formula do integrator)
                 agent_action_bounds: tuple[int, int] = (-1, 1),                       # Clip para PID e TD3_DDPG (formula do integrator)
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

        pyTorch.manual_seed(seed)
        np.random.seed(seed)

        if pid_controller is not None:
            self.pid_controller = pid_controller
        else:
            self.pid_controller = PIDController(Kp, Ki, Kd, 
                                                integrator_bounds, 
                                                sample_period, 
                                                env.unwrapped.error_formula,
                                                controller_type = pid_type
                                                )
            # Used inside ANN class to test PID inside neural network
            batch_PIDController = Batch_PIDController(Kp, Ki, Kd, 
                                                      integrator_bounds, 
                                                      sample_period, 
                                                      pid_type
                                                      )
            # self.batch_pid_controller = batch_PIDController.forward
            
        self.env = env
        self.scheduller = scheduller
        self.ensemble = ensemble_generator
        self.ensemble_size = ensemble_size
        self.tracked_point_name = tracked_point_name
        self.episodes_per_sample = episodes_per_sample
        self.sample_period = sample_period
        self.logs_folder_path = logs_folder_path


        # NOTE: Ao criar uma nova TerminationRule, é necessário fazer um novo elif que calcula o steps_per_episode em função dessa nova regra.
        if env.unwrapped.max_step is not None:
            self.steps_per_episode = env.unwrapped.max_step
        else:
            self.steps_per_episode = env.unwrapped.scheduller.intervals_sum


        if buffer_size is None:
            # n_envs = buffer size, uma vez que num_envs = 1
            buffer_size = self.steps_per_episode * ensemble_size * episodes_per_sample
                          
        self.buffer_size = buffer_size
        
        self.device = pyTorch.device("cuda" if use_GPU else "cpu")
        
        q_lr = mu_lr # TODO: remove rdepois
        self.td3_ddpg = TD3_Agent(
            env.observation_space.shape[0],
            env.action_space.shape[0],
            agent_action_bounds,
            mu_lr, 
            q_lr, 
            target_decay=target_decay, 
            hidden_sizes=[neurons_per_layer, neurons_per_layer, neurons_per_layer],
            pid = batch_PIDController,
            env_name = env_name,
            is_divided = divide_neural_network,
            activation_function=activation_function_name,
            use_activation_func_in_last_layer=use_activation_func_in_last_layer
        )

        # Experience replay memory
        self.replay_buffer = ReplayBuffer(
           obs_dim=env.observation_space.shape[0], 
           act_dim=env.action_space.shape[0], 
           size=self.steps_per_episode# horizon
        )
        self.batch_size = batch_size
        self.epochs = epochs
        self.policy_update_freq = policy_update_freq
        self.noise = noise
        self.noise_clip = noise_clip
        self.discount = discount


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

        records = [] # (*x_vector, y_ref, z_t, PID_action, agent_action, action, reward, error, steps_in_episode)
        returns = []
        
        # steps_extra_info: list[tuple] = []
        steps_in_episode = 0
        total_steps_counter = 0
        count_trainings = 0

        device = self.device

        steps_per_iteration = self.steps_per_episode * self.ensemble_size * self.episodes_per_sample
        iterations: int = math.ceil(steps_to_run / steps_per_iteration)
        # [uncomment] print(f"Steps requested: {steps_to_run}, Steps to be executed: {iterations * steps_per_iteration} (Iterations: {iterations})")

        for iteration in range(1, iterations+1):

            for m in range(self.ensemble_size):
                
                for j in range(self.episodes_per_sample):
                    sample_parameters = self.ensemble.generate_sample()
                    self.pid_controller.reset()
                    self.td3_ddpg.reset_pid()
                    obs, truncated = self.env.reset(options = {"ensemble_sample_parameters": sample_parameters})
                    # print(f"reset OBS @@@ {obs=} | {self.env.unwrapped.error=}")
                    done = False # done is updated by termination rule
                    steps_in_episode = 0
                    episode_reward = 0

                    while not done:

                        pi_action = self.pid_controller(self.env.unwrapped.error)
                
                        agent_action = self.td3_ddpg.select_action(obs, self.noise, self.noise_clip)
                        
                        action = agent_action.item()
                        # action = pi_action + agent_action.item()

                        next_obs, reward, done, truncated, info = self.env.step(action)
                        # print(f"NEXT OBS @@@ {next_obs=} | {self.env.unwrapped.error=}")
                        # input(">>>")
                        steps_in_episode += 1
                        total_steps_counter += 1
                        episode_reward += reward

                        self.replay_buffer.store(obs, action, reward, next_obs, done)
                        records.append((*next_obs, pi_action, agent_action.item(), action, reward, self.env.unwrapped.error, steps_in_episode))

                        obs = next_obs # Can update obs after storing in buffer
                    
                    # Perform the updates
                    qloss_l, mloss_l = self.td3_ddpg.update(
                                        self.replay_buffer, 
                                        self.epochs,
                                        self.batch_size, # self.steps_per_episode, 
                                        self.discount, 
                                        policy_noise=self.noise, 
                                        noise_clip=self.noise_clip, 
                                        policy_freq=self.policy_update_freq
                                    )
                    self.replay_buffer.reset()
                    # q_losses.extend(qloss_l)
                    # mu_losses.extend(mloss_l)
                    count_trainings += 1

                    # End of while loop / end of episode run ###################

                    # [uncomment] print(f"last_steps_in_episode={steps_in_episode}")
                    # [uncomment] last_steps_in_episode = steps_in_episode
                    last_episode_reward = episode_reward # Show how well the trained model is performing
                    returns.append(episode_reward)


        # [uncomment] counter_divided_by_iterations = total_steps_counter/iterations
        # [uncomment] is_same_as_buffer_size = counter_divided_by_iterations == self.buffer_size
        # [uncomment] print(f"{self.buffer_size=}")
        # [uncomment] print(f"{total_steps_counter=} divided by {iterations=} is equal to {counter_divided_by_iterations} ({is_same_as_buffer_size=})")

        if (should_save_records):
            pd.DataFrame(
                records, 
                columns=[f"x{i+1}" for i in range(self.env.unwrapped.x_size)] + \
                        ["y_ref", "z_t", "PID_action", "agent_action", "combined_action", "reward", "error", "steps_in_episode"]
            ).to_csv(
                f"{self.logs_folder_path}/records.csv", 
                index=False
            )
            
            pd.DataFrame(
                returns, 
                columns=["return_per_episode"]
            ).to_csv(
                f"{self.logs_folder_path}/returns.csv", 
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
            self.td3_ddpg.save(self.logs_folder_path)
        
        if (extra_record_only_pid and should_save_records):
            self.record_pid_episode(self.pid_controller, sample_parameters, self.logs_folder_path)

        if (extra_record_only_agent and should_save_records):
            self.record_agent_episode(sample_parameters)

        return last_episode_reward


    def record_pid_episode(self, 
                           pid: Callable[[float], float],
                           sample_parameters: dict[str, float | int | np.ndarray], 
                           logs_folder_path: Optional[str] = None
                           ) -> None:
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
            pi_action = pid(self.env.unwrapped.error)
            next_obs, reward, done, truncated, info = self.env.step(pi_action) 
            steps_in_episode += 1
            records.append((*obs, pi_action, reward, self.env.unwrapped.error, steps_in_episode))

            obs = next_obs # Can update obs after storing in buffer
        
        pd.DataFrame(
            records, 
            columns=[f"x{i+1}" for i in range(self.env.unwrapped.x_size)] + \
                    ["y_ref", "z_t", "PID_action", "reward", "error", "steps_in_episode"]
        ).to_csv(
            f"{logs_folder_path}/only_pid_records.csv", 
            index=False
        )


    def record_agent_episode(self, 
                            sample_parameters: dict[str, float | int | np.ndarray], 
                            policy_weights: Optional[dict[str, any]] = None
                            ) -> None:
        """
        Records the PPO without PID controller response in a single episode and saves it in a csv file.
        """
        for i in range(10):
            records = []
            done = False
            steps_in_episode = 0
            obs, truncated = self.env.reset(options = {"ensemble_sample_parameters": sample_parameters})
            self.pid_controller.reset()
            self.td3_ddpg.reset_pid()
            rewards = 0

            # override ppo weights
            if policy_weights is not None:
                self.td3_ddpg.load(policy_weights)

            # Execute one episode
            while not done:
                pid_action = self.pid_controller(self.env.unwrapped.error)
                agent_action = self.td3_ddpg.select_action(obs, 0, None)
                action = pid_action + agent_action.item()

                next_obs, reward, done, truncated, info = self.env.step(action) 
                steps_in_episode += 1
                rewards += reward

                records.append((*obs, pid_action, agent_action, action, reward, rewards, self.env.unwrapped.error, steps_in_episode))

                obs = next_obs
            
            pd.DataFrame(
                records, 
                columns=[f"x{i+1}" for i in range(self.env.unwrapped.x_size)] + \
                        ["y_ref", "z_t", "PID_action", "agent_action", "combined_action", "reward", "rewards", "error", "steps_in_episode"]
            ).to_csv(
                f"{self.logs_folder_path}/trained_agent_records-{i}.csv", 
                index=False
            )
        

        