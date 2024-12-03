import numpy as np

from environments.CPAP_env import Lung, Ventilator


def expiratory_flow(lung, ventilator, last_pressuse, start_time, current_time):
    _t = current_time - start_time
    _rc = lung._r_aw * lung.c_rs
    return (ventilator.peep - last_pressuse) / lung._r_aw * np.exp(-_t / _rc)


""" Variáveis
TODO
"""
def simulation_model(
        u_t, current_flow, current_volume, current_pressure, 
        /, *, 
        rp, c, rl, # Pacient
        tb, kb, # Blower
        r_aw, c_rs, # Lung
        v_t, peep, rr, t_i, t_ip, # Ventilator
        dt, f_s, t, p, _f, f, v, phase, phase_counter, start_phase_time, i, statefull_obj
        ) -> dict[str, np.float64]:

    lung = Lung(r_aw, c_rs)
    ventilator = Ventilator(v_t, peep, rr, t_i, t_ip)

    if phase == 'exhale':
        if phase_counter == 0:
            start_phase_time = t[i]
        _f[i] = expiratory_flow(lung, ventilator, current_pressure, start_phase_time, t[i])
        
        # v[i] = v[i - 1] + _f[i] * dt
        v[i] = v[i-1] + _f[i] * dt

        # p[i] = f[i] * _r_aw +  v[i] / c_rs  +  peep
        p[i] = _f[i] * lung._r_aw + \
            v[i] / lung.c_rs + \
            ventilator.peep

        phase_counter += 1
        if (phase_counter + 1) >= ventilator.t_e * f_s:
            phase = 'inhale'
            phase_counter = 0


    elif phase == 'inhale':
        # _f[i] = ventilator._f_i
        _f[i] = ventilator._f_i

        # v[i] = v[i-1] + (_f[i] * dt)
        if i > 0:
            v[i] = v[i-1] + (_f[i] * dt)
        else:
            v[i] = 0

        # p[i] = _f[i] x r_aw + v[i] / c_rs  +  peep
        p[i] = _f[i] * lung._r_aw + \
            v[i] / lung.c_rs + \
            ventilator.peep

        phase_counter += 1
        if (phase_counter + 1) >= ventilator.t_i * f_s:
            phase = 'pause'
            phase_counter = 0

    elif phase == 'pause':
        # F = 0
        # V[i] = V[i-1] + (F[i] * dt)
        # P = F x R  +  V x E  +  PEEP
        _f[i] = 0
        v[i] = v[i-1] + (_f[i] * dt)
        p[i] = \
            lung._r_aw * _f[i] + \
            v[i] / lung.c_rs + \
            ventilator.peep
            
        phase_counter += 1
        if (phase_counter + 1) >= ventilator.t_ip * f_s:
            phase = 'exhale'
            phase_counter = 0
            last_pressuse = p[i]
    
    i += 1

    return {
        "x1": f[i-1], 
        "x2": v[i-1],
        "x3": p[i-1] # last_pressuse,
    }
