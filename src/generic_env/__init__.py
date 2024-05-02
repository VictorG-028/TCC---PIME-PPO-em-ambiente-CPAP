from .set_point_env import SetPointEnv

from gymnasium.envs.registration import register

__all__ = [SetPointEnv]

register("SetPointEnv-V0", entry_point="generic_env.set_point_env:SetPointEnv")
