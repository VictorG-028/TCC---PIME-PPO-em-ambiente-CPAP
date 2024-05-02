from algorithms.PIME_PPO import PIME_PPO
# from CustomEnv.WaterTankEnv import WaterTankENv
# from CustomEnv.PHEnv import PHENv
from generic_env import SetPointEnv

import gymnasium
import numpy as np
from stable_baselines3 import PPO

from base.Ensemble import Ensemble
from base.Enums import NamedPoint
from base.Scheduller import Scheduller

"""
l1_t, l2_t -> Nível da água;
a1, a2 -> Áreas dos buracos;
A1, A2 -> Áreas da seção transversal;
K_pump -> Constante da bomba d'água;
u_t -> Ação; Voltagem aplicada a bomba d'água; 
"""
def simulation_model(u_t, l1_t, l2_t, /, g, p1, p2, p3) -> dict[str, np.float64]:
    delta_l1_t = -p1 * np.sqrt(2 * g * l1_t) + p3 * u_t
    l1_t = max([0, l1_t + delta_l1_t]) # TODO: revisar qual valor de l1_t deve entrar na formula delta_l2_t
    delta_l2_t = p1 * np.sqrt(2 * g * l1_t) - p2 * np.sqrt(2 * g * l2_t)
    return {"x1": l1_t, "x2": max([0, l2_t + delta_l2_t])}


def create_environment() -> SetPointEnv:
    set_points = [5, 15, 10]
    intervals = [5, 5, 5]
    scheduller = Scheduller(set_points, intervals) 
    # parameter_names = ["g", "p1", "p2", "p3"]
    distribtions = {
        "g": ("constant", {"constant": 981}),                # g (gravity)
        "p1": ("uniform", {"low": 0.0015, "high": 0.0024}),  # p1
        "p2": ("uniform", {"low": 0.0015, "high": 0.0024}),  # p2
        "p3": ("uniform", {"low": 0.07, "high": 0.17})       # p3
    }
    seed = 42
    params_ensemble = Ensemble(1, distribtions, seed)

    env = gymnasium.make("SetPointEnv-V0", 
                   scheduller       = scheduller, 
                   params_ensemble  = params_ensemble, 
                   simulation_model = simulation_model,
                   start_points     = [20, 20],
                   tracked_point    = NamedPoint.X2
                   )
    
    scheduller = Scheduller(set_points, intervals) 
    return env, scheduller, params_ensemble




env, scheduller, params_ensemble = create_environment()

pime_ppo_controller = PIME_PPO(env, scheduller, 
                               tracked_point_name = NamedPoint.X2)

pime_ppo_controller.guide_with_pid(n_episodes=10)

"""

# Após a orientação inicial, continue o treinamento normal do PPO
ppo_controller.learn(total_timesteps=100)

# Salvar o ppo_controller treinado
ppo_controller.save('PIME-PPO_test')
"""
