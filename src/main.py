from algorithms.PIME_PPO import PIME_PPO
from environments.CPAP_env import CpapEnv
from environments.cascade_water_tank_env import CascadeWaterTankEnv
from environments.ph_control_env import PhControl


experiments = {
    'double_water_tank': {
        'create_env_function': CascadeWaterTankEnv.create_water_tank_environment,
        'logs_folder_path': "logs/ppo/double_water_tank",
        'tracked_point': 'x2',
    },
    'ph_control': {
        'create_env_function': PhControl.create_ph_control_environment,
        'logs_folder_path': "logs/ppo/ph_control",
        'tracked_point': 'x2',
    },
    'CPAP': {
        'create_env_function': CpapEnv.create_cpap_environment,
        'logs_folder_path': "logs/ppo/CPAP",
        'tracked_point': 'x3',
    },
}

create_env_function, logs_folder_path, tracked_point = experiments['double_water_tank'].values()
env, scheduller, ensemble, trained_pid, pid_optimized_params = create_env_function()


pime_ppo_controller = PIME_PPO(
                            env, 
                            scheduller, 
                            ensemble, 
                            # trained_pid,
                            **pid_optimized_params,
                            tracked_point_name=tracked_point,
                            logs_folder_path=logs_folder_path,
                            )

pime_ppo_controller.train(steps_to_run = 100)
