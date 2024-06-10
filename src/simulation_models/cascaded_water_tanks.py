import numpy as np

""" Variáveis
l1_t, l2_t -> Nível da água;
a1, a2 -> Áreas dos buracos;
A1, A2 -> Áreas da seção transversal;
K_pump -> Constante da bomba d'água;
g -> Aceleração gravitacional;
p1 -> (a1 / A1);
p2 -> (a2 / A2);
p3 -> (K_pump / A1);
u_t -> Ação; Voltagem aplicada a bomba d'água; 
"""
def simulation_model(u_t, l1_t, l2_t, /, g, p1, p2, p3) -> dict[str, np.float64]:
    delta_l1_t = -p1 * np.sqrt(2 * g * l1_t) + p3 * u_t
    delta_l2_t = p1 * np.sqrt(2 * g * l1_t) - p2 * np.sqrt(2 * g * l2_t)
    
    # Evitar valores negativos
    l1_t = max([0, l1_t + delta_l1_t]) # TODO: revisar qual valor de l1_t deve entrar na formula delta_l2_t
    l2_t = max([0, l2_t + delta_l2_t])

    return {"x1": l1_t, "x2": l2_t}
