from typing import Callable, Literal

import numpy as np
import matplotlib.pyplot as plt
import control as ct
from scipy.optimize import minimize
import torch as pyTorch

from enums.ErrorFormula import ErrorFormula


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
            controller_type: Literal["PID", "PI", "P"] = "PI",
        ) -> None:
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.min = integrator_bounds[0]
        self.max = integrator_bounds[1]
        self.dt = dt
        self.error_formula = error_formula
        self.previous_error = 0
        self.integral = 0 # precisa ser guardado no self para ir acumulando a cada chamada de controle do pid

        def _PID_formula(error: float) -> float:
            self.integral += error * self.dt
            self.integral = np.clip(self.integral, self.min, self.max)
            derivative = (error - self.previous_error) / self.dt
            self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        def _PI_formula(error: float) -> float:
            assert not isinstance(error, pyTorch.Tensor), "error must be a float, not a pyTorch tensor."
            self.integral += error * self.dt
            self.integral = np.clip(self.integral, self.min, self.max)
            return self.Kp * error + self.Ki * self.integral
        
        def _P_formula(error: float) -> float:
            return self.Kp * error
        
        map_type_to_formula = {
            "PID": _PID_formula,
            "PI": _PI_formula,
            "P": _P_formula
        }
        self.formula = map_type_to_formula[controller_type]

    def __call__(self, error: float) -> float:
        output = self.formula(error)
        self.previous_error = error

        return output
    

    @staticmethod
    def train_pid_with_ZN_method(
                            plant: ct.TransferFunction, 
                            pid_type: Literal["P", "PI", "PID"] = "PI",
                            plot: bool = False,
                        ) -> tuple[float, float, float]:
        """
            Find PID gains using the Ziegler-Nichols reaction curve method.
        """
        # step_y is the height of the step over time
        # plany_y is the height of the output with the respective step_y as input
        time, plant_y = ct.step_response(plant)
        step_y = np.ones_like(time)
        info = ct.step_info(plant)

        # Calculando primeira derivada (dy/dt) e segunda derivada (d²y/dt²)
        dy_dt = np.gradient(plant_y, time)
        d2y_dt2 = np.gradient(dy_dt, time)

        # Encontrando o ponto de inflexão (mudança de sinal na segunda derivada)
        inflexion_idx = np.where(np.diff(np.sign(d2y_dt2)))[0][0]
        # inflexion_idx = 15
        t_inflexion = time[inflexion_idx]    # x
        y_inflexion = plant_y[inflexion_idx] # y
        slope = dy_dt[inflexion_idx]         # Derivada no ponto

        # Interseção da tangente com y=0
        t_intersect_y0 = t_inflexion - y_inflexion / slope

        # Interseção da tangente com y=h (escolha o valor de h)
        h: float = info["Peak"]  # Valor mais alto
        t_intersect_yh = t_inflexion + (h - y_inflexion) / slope

        L = t_intersect_y0 
        T = t_intersect_yh - t_intersect_y0

        if (pid_type == "P"):
            kp = T / L
            ki = 0
            kd = 0
        elif (pid_type == "PI"):
            kp = 0.9 * (T / L)
            ki = L / 0.3
            kd = 0
        elif (pid_type == "PID"):
            kp = 1.2 * (T / L)
            ki = 2 * L
            kd = 0.5 * L
        else:
            raise ValueError("Invalid PID type. Choose between 'P', 'PI' or 'PID'.")

        if (plot):
            
            def tangente(t: float) -> float:
                """tangente(t) = f(t) = y = f'(t_0) ⋅ (t - t_0) + f(t_0)
                - t_0 e y_inflexion=f(t_0) são o tempo e valor no ponto de inflexão.
                - f'(t_0) é o valor da derivada no ponto de inflexão (dy/dt)
                """
                return slope * (t - t_inflexion) + y_inflexion

            
            tangent_x = np.linspace(t_intersect_y0, t_intersect_yh, 1000) # time[-1] = info["PeakTime"]
            tangent_y = tangente(tangent_x)

            plt.plot(time, step_y, linestyle='--', color='black', label="Step inputs")
            plt.plot(tangent_x, tangent_y, color='red', linestyle='--', label="Tangent line")
            plt.plot(time, plant_y, color='blue', label="Plant output")
            plt.axhline(y=h, color="#b0b0b0", linestyle='-', label=f"High point: h={h:.2f}")
            plt.axhline(y=0, color="#b0b0b0", linestyle='-')

            # plt.text(L / 2, -0.1, f"T={T:.2f}", color='black', fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
            # plt.text(T / 2, -0.1, f"L={L:.2f}", color='black', fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))
            plt.text(L, -0.03, f"L", color='black', fontsize=8, ha='center', va='center')
            plt.text(T+L, h+0.03, f"T", color='black', fontsize=8, ha='center', va='center')
            
            # current_ticks = plt.xticks()[0]
            # plt.xticks(list(current_ticks) + [L] + [L+T])

            plt.title("Resposta ao Degrau")
            plt.xlabel("Tempo (s)")
            plt.ylabel("Saída")
            plt.legend()
            plt.grid()
            plt.show()

            exit()

        # Other ZN method (unused)
        # kp = 100000
        
        # untuned_pid = ct.TransferFunction([kp], [1])
        # closed_loop = ct.feedback(untuned_pid * water_tank_model)
        # t, y = ct.impulse_response(closed_loop)

        # # Plot da resposta ao degrau
        # plt.plot(t, y)
        # plt.title(f"Resposta ao Inpulso {kp=}")
        # plt.xlabel("Tempo (s)")
        # plt.ylabel("Saída")
        # plt.grid()
        # plt.show()

        return kp, ki, kd


    @staticmethod
    def train_pid_controller(
                        plant: ct.TransferFunction, 
                        pid_training_method: str | Literal["ZN", "BFGS"] = "BFGS", 
                        pid_type: Literal["P", "PI", "PID"] = "PI",
                        initial_kp = 0, 
                        initial_ki = 0, 
                        initial_kd = 0
                    ) -> Callable[[float], float]:
        """
        Train a PID controller for a given plant using a specified optimization method.

        #### Parameters:
        plant (Callable[[np.ndarray], np.ndarray]): The plant to be controlled.
        pid_type (Literal["P", "PI", "PID"]): The type of PID controller to train.
        pid_training_method (str): The optimization method to use for training the PID controller.

        #### Returns:
        function: A function that takes an input array and returns the control signal from the trained PID controller.
        """

        if (pid_training_method == "ZN"):
            optimized_Kp, optimized_Ki, optimized_Kd = PIDController.train_pid_with_ZN_method(plant, pid_type)
            print(f"ZN results -> {optimized_Kp=}, {optimized_Ki=}, {optimized_Kd=}")
        else:
            def optimize_pid(params):
                Kp, Ki, Kd = params
                pid = ct.TransferFunction([Kp, Ki, Kd], [1, 0.001, 0.001])
                closed_loop = ct.feedback(pid * plant, 1)
                t, y = ct.step_response(closed_loop, T=np.linspace(0, 10, 1000))
                # Objective: minimize the integral of the absolute error
                error = 1 - y  # assuming unit step input
                return np.sum(np.abs(error))

            # Perform pid optimization
            optimized_params = minimize(
                optimize_pid, 
                [initial_kp, initial_ki, initial_kd], 
                method = pid_training_method
            ) 

            optimized_Kp, optimized_Ki, optimized_Kd = optimized_params.x
        

        optimized_pid = ct.TransferFunction(
            [optimized_Kd, optimized_Kp, optimized_Ki], 
            [1, 0.001, 0.001]
        )
        # print(f"Numerador: {optimized_pid.num}")
        # print(f"Denominador: {optimized_pid.den}")
        # print(f"{optimized_pid=}")
        # print(f"{optimized_pid.damp()=}")

        def pid_controller(error: float) -> float:
            """
            Compute the control signal using the trained PID controller.
            """
            pid_action = ct.forced_response(
                optimized_pid, 
                T = np.array([0, 1e-6]), # 1e-6 para intervalos instantâneos
                # T = np.array([0, 1]),  # 1 para intervalos longos
                U = np.array([0, error])
            )

            return pid_action.y[0, 1]  # Retornar o último valor da resposta

        return pid_controller, {'optimized_Kp': optimized_Kp, 'optimized_Ki': optimized_Ki, 'optimized_Kd': optimized_Kd}



class Batch_PIDController:
    """
    PID controller for a discrete-time state space model (dt = 1).
    This controller is designed to receive optimized Kp, Ki, and Kd values.
    Now supports batch processing of multiple errors using PyTorch tensors.
    """
    def __init__(
            self, 
            Kp, Ki, Kd, 
            integrator_bounds: list[int, int],
            dt=1,
            error_formula: ErrorFormula | Callable[[float, float], float] = ErrorFormula.DIFFERENCE, 
            controller_type: Literal["PID", "PI", "P"] = "PI",
        ) -> None:
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.min = integrator_bounds[0]
        self.max = integrator_bounds[1]
        self.dt = dt
        self.error_formula = error_formula

        # Inicializar como tensor vazio, pois o tamanho do batch pode variar
        self.previous_error = None
        self.integral = None

        def _PID_formula(error: pyTorch.Tensor) -> pyTorch.Tensor:
            """PID formula for batch processing"""
            self.integral += error * self.dt
            self.integral = pyTorch.clamp(self.integral, self.min, self.max)
            derivative = (error - self.previous_error) / self.dt
            return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        def _PI_formula(error: pyTorch.Tensor) -> pyTorch.Tensor:
            """PI formula for batch processing"""
            self.integral += error * self.dt
            self.integral = pyTorch.clamp(self.integral, self.min, self.max)
            return self.Kp * error + self.Ki * self.integral
        
        def _P_formula(error: pyTorch.Tensor) -> pyTorch.Tensor:
            """P formula for batch processing"""
            return self.Kp * error
        
        map_type_to_formula = {
            "PID": _PID_formula,
            "PI": _PI_formula,
            "P": _P_formula
        }
        self.formula = map_type_to_formula[controller_type]

    def __call__(self, error: pyTorch.Tensor) -> pyTorch.Tensor:
        """Process batch of errors using the selected PID formula"""
        if self.previous_error is None:
            # Inicializar previous_error e integral no mesmo formato de error
            self.previous_error = pyTorch.zeros_like(error)
            self.integral = pyTorch.zeros_like(error)

        output = self.formula(error)
        self.previous_error = error.clone()  # Atualiza previous_error corretamente
        return output
