import gymnasium
import numpy as np
from stable_baselines3 import PPO

from base.Enums import NamedPoint
from base.Scheduller import Scheduller

class PIDController:
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
        self.timstep = 0
        self.scheduller = scheduller
        self.set_point = scheduller.get_set_point_at(step=0)
        self.integral = 0


    def update(self, observation, timestep):
        # TODO: Ver se PID deve usar um timestep interno ou receber de input
        # TODO: Ver se é melhor colocar um step interno no scheduller e forçar chamar o get apenas uma vez por step
        # TODO: Ver se o PID se autoregula (modifica kp, ki) ou apenas retorna uma ação
        set_point = self.scheduller.get_set_point_at(step=timestep)

        # TODO: Ver se esse if que reseta a integral deve existir
        if (self.set_point != set_point):
            self.integral = 0
            self.set_point = set_point

        self.timstep += 1

        error = set_point - observation
        self.integral += error
        action = -self.kp * error + self.ki * self.integral
        return action
    

    def reset(self):
        self.timstep = 0


class PIME_PPO:
    """An algorithm made with PPO."""

    def __init__(self, 
                 env: gymnasium.Env,
                 sheduller: Scheduller,
                 verbose: int = 1,
                 kp: np.float64 = 0.1, 
                 ki: np.float64 = 0.1,
                 tracked_point_name: NamedPoint = NamedPoint.X1
                 ) -> None:
        self.env = env
        self.sheduller = sheduller
        self.ppo = PPO('MultiInputPolicy', env, verbose=verbose) # "MlpPolicy"
        self.pid_controller = PIDController(kp, ki, sheduller)
        self.tracked_point_name = tracked_point_name


    def guide_with_pid(self, n_episodes = 10) -> None:
        
        for episode in range(n_episodes):
            obs, truncated = self.env.reset()
            done = False
            i = 0
            while not done:
                print("returned obs: ", obs)
                action = self.pid_controller.update(
                    obs[self.tracked_point_name.value],
                    self.env.unwrapped.timestep
                )

                # TODO: estudar e implementar discretização com método de Euler (referência 32 do paper)
                # action = np.clip(int(action), 0, env.action_space.n - 1)

                obs, reward, done, truncated = self.env.step(action)

                # TODO fazer PPO aprender com base no PID
                # Aqui você pode adicionar a lógica para atualizar o modelo PPO com esta ação
                i += 1
                if i == 15:
                    exit()

