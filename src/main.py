from algorithms.PIME_PPO import PIME_PPO
# from CustomEnv.WaterTankEnv import WaterTankENv
# from CustomEnv.PHEnv import PHENv
from algorithms.wrappers.DictToArray import DictToArrayWrapper
from enums.TerminationRule import TerminationRule
from enums.ErrorFormula import ErrorFormula

from modules.Ensemble import Ensemble
from modules.Scheduller import Scheduller

from set_point_env import SetPointEnv

import gymnasium
import numpy as np
from stable_baselines3 import PPO

from simulation_models.cascaded_water_tanks import simulation_model as double_tank_simultaion
from simulation_models.ph_residual_water_treatment import simulation_model as ph_simulation


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
    params_ensemble = Ensemble(10, distribtions, seed)

    env = gymnasium.make("SetPointEnv-V0", 
                   scheduller       = scheduller, 
                   simulation_model = double_tank_simultaion,
                   ensemble_params  = params_ensemble.get_param_set(), 
                   termination_rule = TerminationRule.INTERVALS,
                   error_formula    = ErrorFormula.DIFFERENCE,
                   start_points     = [20, 20],
                   tracked_point    = 'x2',
                   )
    env = DictToArrayWrapper(env)
    
    scheduller = Scheduller(set_points, intervals) 
    return env, scheduller, params_ensemble



env, scheduller, params_ensemble = create_environment()

pime_ppo_controller = PIME_PPO(env, scheduller, params_ensemble, 
                               tracked_point_name = 'x2')

pime_ppo_controller.train(n_episodes = 10)
"""
pime_ppo_controller.guide_with_pid(n_episodes=10)

# Após a orientação inicial, continue o treinamento normal do PPO
ppo_controller.learn(total_timesteps=100)

# Salvar o ppo_controller treinado
ppo_controller.save('PIME-PPO_test')
"""
