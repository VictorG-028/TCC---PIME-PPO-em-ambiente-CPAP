from .base_set_point_env import BaseSetPointEnv
from .cascade_water_tank_env import CascadeWaterTankEnv
from .ph_control_env import PhControl
from .CPAP_env import CpapEnv

from gymnasium.envs.registration import register

__all__ = [base_set_point_env]

register("BaseSetPointEnv-V0", entry_point="environments.base_set_point_env:BaseSetPointEnv")
register("CascadeWaterTankEnv-V0", entry_point="environments.cascade_water_tank_env:CascadeWaterTankEnv")
register("PhControl-V0", entry_point="environments.residual_water_treatment_env:PhControl")
register("CpapEnv-V0", entry_point="environments.CPAP_env:CpapEnv")
