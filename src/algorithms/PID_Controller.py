from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

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
            use_derivative: bool = False
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

        def _PID_formula(error):
            self.integral += np.clip(error * self.dt, self.min, self.max)
            derivative = (error - self.previous_error) / self.dt
            self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        def _PI_formula(error):
            self.integral += np.clip(error * self.dt, self.min, self.max)
            return self.Kp * error + self.Ki * self.integral
        
        self.formula = _PID_formula if use_derivative else _PI_formula

    def __call__(self, error: float) -> float:
        output = self.formula(error)
        self.previous_error = error

        return output
    
    def tunnig_wiht_Ziegler_Nichols_method(self, Kp_critical: float, L_critical: float):
        """
        - [wiki](https://en.wikipedia.org/wiki/Ziegler%E2%80%93Nichols_method)
        - [tutorial](https://aleksandarhaber.com/model-assisted-ziegler-nichols-pid-control-tuning-method/)
        
        Perguntas para chat gpt:
        - Qual o tempo curto suficiente para formar um inpulso / short pulse
        - Qual o tempo do inpulso, mostra um código python com uma implementação de inpulso
        Lembrar:
        - Usar como argumento no paper: o método zieger-nichols gera overshot agressivo  e o PPO atenua esse overshot
        
        [en]
        In Ziegler Wichols method, input = y_target = inpulse /\ or short pulse _|▔|_
        complete model = plant = complete system = [y_target -> [PID] * [model_transfer_fn] -> y]
        kp_critical = Kp_ultimate = Kp value such that the system (plant, model, PID * model_transfer_fn) oscillates at a constant amplitude
        L_critical = period of oscillation when using kp_critical = time between two peaks or two valleys in the output curve when using kp_critical
        
        [pt-br]
        y = y_out = saída da planta = curva com ocilações = curva com picos e vales e 
        No método de Ziegler Wichols, input =  y_objetivo = inpulso /\ ou pulso curto _|▔|_
        modelo completo = planta = sistema completo = [y_objetivo -> [PID] * [fn_transferência_do_modelo] -> y]
        Kp_critical = Kp_ultimate = valor de Kp tal que o modelo completo oscila com amplitude constante
        L_critical = período de oscilação ao usar Kp_critical = tempo entre dois picos ou dois vales na curva de saída ao usar Kp_critical
        """

        

        self.Kp = 0.6 * Kp_critical 
        self.Ki = 1.2 * Kp_critical / L_critical
        self.Kd = 0.075 * Kp_critical * L_critical




"""
# https://www.google.com/search?q=Ziegler%E2%80%93Nichols+pid+training+python&sca_esv=076c6e27218f3cb7&sxsrf=ADLYWIKsBw1zCoxUFKFnQBUmIDX8b_9ogA%3A1733761502763&ei=3hlXZ6uaLoTb5OUPmfaxkAs&ved=0ahUKEwirgf3zjJuKAxWELbkGHRl7DLIQ4dUDCA8&uact=5&oq=Ziegler%E2%80%93Nichols+pid+training+python&gs_lp=Egxnd3Mtd2l6LXNlcnAiJVppZWdsZXLigJNOaWNob2xzIHBpZCB0cmFpbmluZyBweXRob24yBRAhGKABMgUQIRigAUiKDlDLAVjODHABeAGQAQCYAesBoAHdCaoBBTAuMi40uAEDyAEA-AEBmAIHoALnCcICChAAGLADGNYEGEeYAwCIBgGQBgiSBwUxLjIuNKAH6xE&sclient=gws-wiz-serp
https://sites.poli.usp.br/d/PME2472/ziegler.pdf
https://github.com/SzymonK1306/Ziegler-Nichols-tuning-method
https://jckantor.github.io/CBE30338/04.12-Interactive-PID-Control-Tuning-with-Ziegler--Nichols.html
https://en.wikipedia.org/wiki/Proportional%E2%80%93integral%E2%80%93derivative_controller
https://www.youtube.com/watch?v=YYxkS1iFdVk&t=5s
https://aleksandarhaber.com/model-assisted-ziegler-nichols-pid-control-tuning-method/#google_vignette
https://python-control.readthedocs.io/en/latest/generated/control.impulse_response.html
https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/pendulum.py
http://appavl.psxsistemas.com.br:882/pergamumweb/vinculos/000030/000030c7.pdf
https://sites.poli.usp.br/d/PME2472/ziegler.pdf
"""

