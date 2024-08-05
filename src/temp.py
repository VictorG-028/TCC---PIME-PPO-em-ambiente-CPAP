import sympy as sp

# Definir variáveis simbólicas
t = sp.symbols('t', real=True, positive=True)
s = sp.symbols('s')
ph_t = sp.Function('pH')(t)
u_t = sp.Function('u')(t)
ph_s = sp.Function('pH')(s)
u_s = sp.Function('u')(s)
p1, K_reagent = sp.symbols('p1 K_reagent')
ph_laplace = sp.symbols('P', cls=sp.Function)(s)

# Equação diferencial
delta_ph = -p1 * ph_t + K_reagent * u_t

# TODO: Estudar o que é essa equação diferencial (o que sp.Eq faz ?)
dph_dt = sp.Derivative(ph_t, t)
diff_eq = sp.Eq(dph_dt, -p1 * ph_t + K_reagent * u_t)

# Transformada de Laplace
diff_eq_transformed = sp.laplace_transform(diff_eq, t, s)
delta_ph_transformed = sp.laplace_transform(delta_ph, t, s)


laplace_eq = diff_eq_transformed.subs({
    sp.laplace_transform(ph_t, t, s)[0]: ph_laplace, 
    sp.laplace_transform(u_t, t, s)[0]: u_s
})
transfer_function_eq = sp.Eq(
    ph_laplace, 
    diff_eq_transformed.rhs / diff_eq_transformed.lhs.coeff(ph_laplace)
)
transfer_function = sp.simplify(transfer_function_eq.rhs)
