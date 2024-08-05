import numpy as np

"""
Modelo de simulação para controle de pH em uma estação de 
tratamento de águas residuais.
"""

""" Glossário de variáveis
ph_t -> pH;
waste_conc_t -> Concentração de resíduos indesejados;
p1 -> ([HCL,C] * q_[HCL,C]) / V -> concentração de ácido clorídrico[HCl,C] vezes constante de "flow" de [HCl,C] dividido por volume do tanque de reação;
p2 -> q_ww / V -> taxa de fluxo de entrada de água residual q_ww dividido pelo volume do tanque de reação V;
K_reagent -> Constante de reagente, descrevendo a eficiência do reagente na neutralização do pH;
u_t -> Ação; Dosagem de reagente;
"""
def simulation_model(u_t, ph_t, hcl_concentration_t, /, *, p1, p2) -> dict[str, np.float64]:

    # Atualiza a concentração de HCl usando o método de Euler
    delta_hcl_t = -p2 * hcl_concentration_t + p1 * u_t

    # Evita valores negativos
    hcl_concentration_t = max(0, hcl_concentration_t + delta_hcl_t) 
    
    # Calcula o pH baseado na nova concentração de H+
    h_concentration_t = hcl_concentration_t  # Simplificação, assumindo [HCl] ≈ [H+]
    
     # Evitar log de zero, assumindo pH máximo de 14
    if h_concentration_t > 0:
        ph_t = -np.log10(h_concentration_t) 
    else:
        ph_t = 14
    
    return {"HCl": hcl_concentration_t, "pH": ph_t}
    
    # delta_ph = -p1 * ph_t + p1 * u_t
    # delta_hcl_conc = -p2 * hcl_conc_t + p1 * u_t

    # # Evitar valores negativos
    # ph_t = max(0, ph_t + delta_ph)
    # hcl_conc_t = max(0, hcl_conc_t + delta_hcl_conc)

    # return {"ph": ph_t, "conc": hcl_conc_t}
