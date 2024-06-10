from .set_point_env import SetPointEnv

from gymnasium.envs.registration import register

__all__ = [set_point_env]

register("SetPointEnv-V0", entry_point="set_point_env.set_point_env:SetPointEnv")
