import numpy as np

"""
Modelo de simulação para controle de pH em uma estação de 
tratamento de águas residuais.
"""

""" Variáveis
ph_t -> pH;
conc_t -> Concentração de resíduos;
g -> Aceleração gravitacional;
k1 -> ;
k2 -> ;
K_reagent -> ;
u_t -> Ação; Dosagem de reagente;
"""
def simulation_model(u_t, ph_t, conc_t, /, g, k1, k2, K_reagent) -> dict[str, np.float64]:

    delta_ph = -k1 * ph_t + K_reagent * u_t
    conc_dot = -k2 * conc_t + K_reagent * u_t

    # Evitar valores negativos
    ph_t = max(0, ph_t + delta_ph)
    conc_t = max(0, conc_t + conc_dot)

    return {"ph": ph_t, "conc": conc_t}
