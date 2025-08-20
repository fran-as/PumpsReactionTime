# app.py
import math
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Memoria de Cálculo – Tiempo de reacción de bombas (VDF)",
    layout="wide"
)

# ------------------------------
# Utilidades
# ------------------------------
DEC = 2
G = 9.81  # m/s^2

def fmt(x, nd=DEC):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)

def v(row, keys, default=0.0):
    """Lee la primera clave disponible en 'row' y devuelve float (o default)."""
    for k in keys:
        if k in row and pd.notna(row[k]) and str(row[k]) != "":
            try:
                return float(row[k])
            except Exception:
                continue
    return float(default)

def get_ratio(row):
    # r = n_motor / n_bomba
    return v(row, ["ratio", "RelacionTransmision", "TransmissionRatio", "relacion", "r"], 1.0)

def get_tag_column(df):
    # Por definición: la primera columna es el TAG
    return df.columns[0]

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# ------------------------------
# Carga de datos
# ------------------------------
DATA_FILE = "bombas_dataset_with_torque_params.xlsx"
df = pd.read_excel(DATA_FILE, sheet_name="dataSet")
col_tag = get_tag_column(df)
tags = df[col_tag].astype(str).tolist()

# ------------------------------
# Sidebar – selección
# ------------------------------
st.sidebar.header("Selección")
tag_sel = st.sidebar.selectbox("TAG", options=tags, index=0)
row = df[df[col_tag].astype(str) == str(tag_sel)].iloc[0]

# ------------------------------
# Sección 1 — Atributos
# ------------------------------
st.markdown("# Memoria de Cálculo – Tiempo de reacción de bombas (VDF)")
st.markdown("### 1) Atributos del equipo y del sistema")

# Motor
P_motor_kW = v(row, ["MotorPowerInstalled_kW", "MotorPower_kW", "P_motor_kW"], 0.0)
J_m        = v(row, ["J_m", "J_motor", "Motor_J_kgm2"], 0.0)
n_m_min    = v(row, ["MotorSpeedMin_rpm", "n_motor_min"], 0.0)
n_m_max    = v(row, ["MotorSpeedMax_rpm", "n_motor_max"], 0.0)
T_nom      = v(row, ["T_nom", "T_motor_nominal", "Par_nominal_Nm"], 0.0)

# Transmisión
ratio = get_ratio(row)  # r = n_motor / n_bomba
J_driver = v(row, ["J_driver", "J_driver_total", "J_polea_motor"], 0.0)
J_driven = v(row, ["J_driven", "J_driven_total", "J_polea_bomba"], 0.0)

# Bomba
D_imp_mm  = v(row, ["Impeller_D_mm", "DiametroImpulsor_mm"], 0.0)
J_imp     = v(row, ["J_imp", "Impeller_J_kgm2", "J_impulsor"], 0.0)
Q_min     = v(row, ["Q_min_m3h"], 0.0)   # m^3/h
Q_max     = v(row, ["Q_max_m3h"], 0.0)   # m^3/h

# Sistema/fluido/hidráulica
H0        = v(row, ["H0_m"], 0.0)
K         = v(row, ["K_m_s2"], 0.0)        # asumido en m/(m^3/h)^2 (m por (m3/h)^-2)
eta_a     = v(row, ["eta_a"], 0.0)
eta_b     = v(row, ["eta_b"], 0.0)
eta_c     = v(row, ["eta_c"], 0.0)
rho       = v(row, ["rho_kgm3"], 1000.0)
n_ref     = v(row, ["n_ref_rpm"], 0.0)

# Derivados
n_p_min = n_m_min / ratio if ratio != 0 else 0.0
n_p_max = n_m_max / ratio if ratio != 0 else 0.0

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("**Motor**")
    st.write(f"Potencia instalada: {fmt(P_motor_kW)} kW")
    st.write(f"Inercia motor $J_m$: {fmt(J_m)} kg·m²")
    st.write(f"Velocidad Motor min–max [rpm]: {fmt(n_m_min)} – {fmt(n_m_max)}")
    st.write(f"Par nominal $T_{{nom}}$: {fmt(T_nom)} Nm")

with c2:
    st.markdown("**Transmisión**")
    st.write(f"Relación $r = n_{{motor}}/n_{{bomba}}$: {fmt(ratio)}")
    st.write(f"Inercia lado motor $J_{{driver}}$: {fmt(J_driver)} kg·m²")
    st.write(f"Inercia lado bomba $J_{{driven}}$: {fmt(J_driven)} kg·m²")

with c3:
    st.markdown("**Bomba**")
    st.write(f"Ø impulsor: {fmt(D_imp_mm)} mm")
    st.write(f"Inercia impulsor $J_{{imp}}$: {fmt(J_imp)} kg·m²")
    st.write(f"Velocidad Bomba min–max [rpm]: {fmt(n_p_min)} – {fmt(n_p_max)}")

with c4:
    st.markdown("**Sistema/Fluido**")
    st.latex(r"H(Q) = H_0 + K\,Q^2")
    st.latex(r"\eta(Q) = \eta_a + \eta_b\,Q + \eta_c\,Q^2")
    st.write(f"H0: {fmt(H0)} m,  K: {fmt(K)} (ver nota de unidades)")
    st.write(f"$\\eta_a$: {fmt(eta_a)}, $\\eta_b$: {fmt(eta_b)}, $\\eta_c$: {fmt(eta_c)}")
    st.write(f"$\\rho$: {fmt(rho)} kg/m³,  $n_{{ref}}$: {fmt(n_ref)} rpm")
    st.write(f"Rango de caudal (dataset): {fmt(Q_min)} – {fmt(Q_max)} m³/h")

st.divider()

# ------------------------------
# Sección 2 — Dinámica inercial (sin efectos hidráulicos)
# ------------------------------
st.markdown("### 2) Dinámica inercial (sin efectos hidráulicos)")

# Entradas compactas
ci, cf, ct, cr = st.columns([1, 1, 1, 1])
with ci:
    n_ini_m = st.number_input("Velocidad Motor inicial [rpm]", value=float(n_m_min), step=1.0, format="%.2f")
with cf:
    n_fin_m = st.number_input("Velocidad Motor final [rpm]", value=float(n_m_max), step=1.0, format="%.2f")
with ct:
    T_disp  = st.number_input("Par disponible T_disp [Nm]", value=float(T_nom), step=1.0, format="%.2f")
with cr:
    rampa_vdf = st.number_input("Rampa VDF (motor) [rpm/s]", value=300.0, min_value=1.0, step=1.0, format="%.2f")

Δn = max(n_fin_m - n_ini_m, 0.0)

# 2.1 – Inercia equivalente (al eje del motor)
st.markdown("**2.1) Inercia equivalente al eje del motor**")
fcol, scol = st.columns(2)
with fcol:
    st.markdown("**(A) Fórmula**")
    st.latex(r"J_{\mathrm{eq}} = J_m + J_{\mathrm{driver}} + \dfrac{J_{\mathrm{driven}} + J_{\mathrm{imp}}}{r^2}")
    st.caption(r"r = \dfrac{n_{\mathrm{motor}}}{n_{\mathrm{bomba}}}")
with scol:
    st.markdown("**(B) Sustitución**")
    J_eq = J_m + J_driver + (J_driven + J_imp)/max(ratio**2, 1e-12)
    st.latex(
        rf"J_{{\mathrm{{eq}}}} = {fmt(J_m)} + {fmt(J_driver)} + \dfrac{{{fmt(J_driven)} + {fmt(J_imp)}}}{{({fmt(ratio)})^2}}"
        rf"\Rightarrow\; {fmt(J_eq)}\ \mathrm{{kg\cdot m^2}}"
    )

st.divider()

# 2.2 – Aceleración por par (sin hidráulica)
st.markdown("**2.2) Aceleración por par disponible**")
fcol, scol = st.columns(2)
with fcol:
    st.markdown("**(A) Fórmula**")
    st.latex(r"\dot{n}_{\mathrm{torque}}=\dfrac{60}{2\pi}\,\dfrac{T_{\mathrm{disp}}}{J_{\mathrm{eq}}}\quad [\mathrm{rpm/s}]")
with scol:
    st.markdown("**(B) Sustitución**")
    n_dot_nohyd = (60.0/(2.0*math.pi)) * T_disp / max(J_eq, 1e-12)
    st.latex(
        rf"\dot{{n}}_{{\mathrm{{torque}}}}=\frac{{60}}{{2\pi}}\cdot\frac{{{fmt(T_disp)}}}{{{fmt(J_eq)}}}"
        rf"= {fmt(n_dot_nohyd)}\ \mathrm{{rpm/s}}"
    )

st.divider()

# 2.3 – Tiempos por par, por rampa y tiempo final
st.markdown("**2.3) Tiempos por par, por rampa y tiempo final**")
fcol, scol = st.columns(2)
with fcol:
    st.markdown("**(A) Fórmulas**")
    st.latex(r"t_{\mathrm{par}}=\dfrac{\Delta n}{\dot{n}_{\mathrm{torque}}},\qquad"
             r"t_{\mathrm{rampa}}=\dfrac{\Delta n}{\mathrm{rampa}_{\mathrm{VDF}}},\qquad"
             r"t_{\mathrm{final}}=\max\!\left(t_{\mathrm{par}},t_{\mathrm{rampa}}\right)")
with scol:
    st.markdown("**(B) Sustitución**")
    t_par   = Δn / max(n_dot_nohyd, 1e-12)
    t_rampa = Δn / max(rampa_vdf, 1e-12)
    t_final_nohyd = max(t_par, t_rampa)
    st.latex(rf"\Delta n = {fmt(Δn)}\ \mathrm{{rpm}}")
    st.latex(rf"t_{{\mathrm{{par}}}}=\frac{{{fmt(Δn)}}}{{{fmt(n_dot_nohyd)}}}= {fmt(t_par)}\ \mathrm{{s}}")
    st.latex(rf"t_{{\mathrm{{rampa}}}}=\frac{{{fmt(Δn)}}}{{{fmt(rampa_vdf)}}}= {fmt(t_rampa)}\ \mathrm{{s}}")
    st.latex(rf"\boxed{{t_{{\mathrm{{final}}}}= {fmt(t_final_nohyd)}\ \mathrm{{s}}}}")

st.caption("En esta sección se considera únicamente la respuesta inercial del tren motriz (sin par hidráulico).")

st.divider()

# ------------------------------
# Sección 3 — Dinámica con carga hidráulica
# ------------------------------
st.markdown("### 3) Dinámica con carga hidráulica")

# (1) Definiciones del modelo hidráulico
fcol, scol = st.columns(2)
with fcol:
    st.markdown("**(A) Leyes usadas**")
    st.latex(r"H(Q)=H_0 + K\,Q^2")
    st.latex(r"\eta(Q)=\eta_a + \eta_b\,Q + \eta_c\,Q^2")
    st.latex(r"Q(n_p)=\alpha\,n_p\quad\text{(afinidad: caudal proporcional a rpm de bomba)}")
    st.caption("Aquí se aproxima $Q$ lineal con $n_p$.")
with scol:
    st.markdown("**(B) Sustitución de parámetros**")
    # elegimos alpha haciendo pasar Q por el origen: alpha = Q_max / n_p_max (m3/h por rpm)
    alpha_m3h_per_rpm = (Q_max / max(n_p_max, 1e-12)) if Q_max > 0 and n_p_max > 0 else 0.0
    st.latex(rf"\alpha=\dfrac{{Q_{{max}}}}{{n_{{p,max}}}}=\dfrac{{{fmt(Q_max)}}}{{{fmt(n_p_max)}}}"
             rf"= {fmt(alpha_m3h_per_rpm)}\ \mathrm{{(m^3/h)/rpm}}")

st.divider()

# (2) Par cargado por hidráulica y reflejado al motor
fcol, scol = st.columns(2)
with fcol:
    st.markdown("**(A) Fórmulas de par de carga**")
    st.latex(r"P_h=\rho\,g\,Q\,H(Q)")
    st.latex(r"P_{shaft}=\dfrac{P_h}{\eta(Q)}")
    st.latex(r"\omega_p=\dfrac{2\pi\,n_p}{60},\qquad T_p=\dfrac{P_{shaft}}{\omega_p}")
    st.latex(r"\text{Al motor:}\quad T_{\mathrm{load,eq}}=\dfrac{T_p}{r}")
    st.caption("Se supone transmisión sin pérdidas (aprox. potencia conservada).")
with scol:
    st.markdown("**(B) Sustitución genérica en función de $n_p$**")
    st.latex(rf"Q(n_p)={fmt(alpha_m3h_per_rpm)}\,n_p\;[\mathrm{{m^3/h}}]")
    st.latex(r"H(Q)=H_0+KQ^2")
    st.latex(r"\eta(Q)=\eta_a+\eta_bQ+\eta_cQ^2")

st.divider()

# (3) Aceleración neta y tiempo sobre un rango de rpm de bomba
st.markdown("**(3.1) Selección del rango de velocidad de la bomba**")

# Slider de rango de rpm de bomba
n1_p_default = max(n_p_min, (n_p_min + n_p_max) / 3.0)
n2_p_default = min(n_p_max, (n_p_min + n_p_max) * 0.8)

n1_p, n2_p = st.slider(
    "Rango de velocidad de bomba [rpm]",
    min_value=float(max(0.0, n_p_min)),
    max_value=float(max(n_p_min, n_p_max)),
    value=(float(n1_p_default), float(n2_p_default)),
    step=1.0
)

# rpm motor asociadas
n1_m = n1_p * ratio
n2_m = n2_p * ratio

cL, cR = st.columns(2)
with cL:
    st.write(f"Velocidad bomba: {fmt(n1_p)} → {fmt(n2_p)} rpm")
with cR:
    st.write(f"Velocidad motor equivalente: {fmt(n1_m)} → {fmt(n2_m)} rpm")

st.markdown("**(3.2) Aceleración neta (con carga hidráulica)**")
fcol, scol = st.columns(2)
with fcol:
    st.markdown("**(A) Fórmula**")
    st.latex(r"\dot{n}_{\mathrm{net}}=\dfrac{60}{2\pi}\,\dfrac{T_{\mathrm{disp}}-T_{\mathrm{load,eq}}(n_p)}{J_{\mathrm{eq}}},"
             r"\qquad n_p=\dfrac{n_m}{r}")
    st.caption("Se limita además por la rampa del VDF: $\,\dot{n}=\min(\dot{n}_{net},\,\mathrm{rampa}_{VDF})$.")
with scol:
    st.markdown("**(B) Sustitución (punto inicial del rango)**")
    # Punto inicial para mostrar sustitución
    Q1_m3h = alpha_m3h_per_rpm * n1_p
    Q1_m3s = Q1_m3h / 3600.0
    H1 = H0 + K * (Q1_m3h**2)  # Ojo: K con Q en m3/h (dataset). Si K fuera en SI, conviértelo aquí.
    eta1 = clamp(eta_a + eta_b*Q1_m3h + eta_c*(Q1_m3h**2), 0.05, 0.95)
    w1 = 2*math.pi*(n1_p)/60.0
    P_h1 = rho * G * Q1_m3s * H1
    P_shaft1 = P_h1 / max(eta1, 1e-6)
    T_p1 = P_shaft1 / max(w1, 1e-9)
    T_eq1 = T_p1 / max(ratio, 1e-12)
    n_dot_net_1 = (60.0/(2.0*math.pi))*(T_disp - T_eq1)/max(J_eq, 1e-12)
    st.latex(rf"Q_1={fmt(Q1_m3h)}\ \mathrm{{m^3/h}},\quad H_1={fmt(H1)}\ \mathrm{{m}},\quad \eta_1={fmt(eta1)}")
    st.latex(rf"P_{{h,1}}=\rho g Q_1 H_1={fmt(P_h1)}\ \mathrm{{W}},\quad P_{{shaft,1}}={fmt(P_shaft1)}\ \mathrm{{W}}")
    st.latex(rf"T_{{p,1}}=\frac{{P_{{shaft,1}}}}{{\omega_1}}={fmt(T_p1)}\ \mathrm{{Nm}},\quad "
             rf"T_{{load,eq,1}}=\frac{{T_{{p,1}}}}{{r}}={fmt(T_eq1)}\ \mathrm{{Nm}}")
    st.latex(rf"\dot{{n}}_{{net,1}}=\frac{{60}}{{2\pi}}\frac{{{fmt(T_disp)}-{fmt(T_eq1)}}}{{{fmt(J_eq)}}}"
             rf"= {fmt(n_dot_net_1)}\ \mathrm{{rpm/s}}")

st.markdown("**(3.3) Tiempo para recorrer el rango seleccionado**")

# Integración numérica simple en rpm de motor (Euler explícito)
# Respetando rampa del VDF y anulando si la carga excede el par disponible
def eta_of_Q(Q_m3h: float) -> float:
    return clamp(eta_a + eta_b*Q_m3h + eta_c*(Q_m3h**2), 0.05, 0.95)

def hydraulic_load_torque_equiv(n_m: float) -> float:
    """Retorna T_load equivalente en el eje del motor (Nm) a una velocidad de motor n_m."""
    n_p = n_m / max(ratio, 1e-12)
    Q_m3h = alpha_m3h_per_rpm * n_p                # m^3/h
    Q_m3s = Q_m3h / 3600.0                         # m^3/s
    H_val = H0 + K*(Q_m3h**2)                      # m
    eta_val = eta_of_Q(Q_m3h)
    w_p = 2*math.pi*n_p/60.0                       # rad/s
    P_h = rho * G * Q_m3s * H_val                  # W
    P_shaft = P_h / max(eta_val, 1e-6)             # W
    T_p = P_shaft / max(w_p, 1e-9)                 # Nm
    return T_p / max(ratio, 1e-12)                 # Nm, reflejado al motor

def integrate_time(n_start_m: float, n_end_m: float, step_rpm: float = 1.0) -> tuple[float, float, bool]:
    """
    Integra el tiempo desde n_start_m hasta n_end_m (rpm de motor) con:
    dn/dt = min( (60/2π)*(T_disp - T_load_eq)/J_eq , rampa_vdf ),  saturado a >= 0.
    Retorna (t_total_s, n_dot_min, reached_flag).
    """
    if n_end_m <= n_start_m + 1e-9:
        return 0.0, 0.0, True

    n = n_start_m
    t_total = 0.0
    n_dot_min = 1e9
    reached = True

    while n < n_end_m - 1e-9:
        T_load_eq = hydraulic_load_torque_equiv(n)
        n_dot_torque = (60.0/(2.0*math.pi))*(T_disp - T_load_eq)/max(J_eq, 1e-12)  # rpm/s
        n_dot = min(n_dot_torque, rampa_vdf)
        if n_dot <= 0.0:   # par insuficiente → no se alcanzará el objetivo
            reached = False
            break
        dn = min(step_rpm, n_end_m - n)
        dt = dn / n_dot
        t_total += dt
        n += dn
        n_dot_min = min(n_dot_min, n_dot)
    return t_total, n_dot_min if n_dot_min < 1e9 else 0.0, reached

# Ejecuta integración
t_total_hyd, n_dot_min_seen, reached_ok = integrate_time(n1_m, n2_m, step_rpm=1.0)

# Mostrar resultados con estilo similar
fcol, scol = st.columns(2)
with fcol:
    st.markdown("**(A) Resultado**")
    if reached_ok:
        st.latex(rf"\boxed{{t_{{\mathrm{{hid}}}}= {fmt(t_total_hyd)}\ \mathrm{{s}}}}")
        st.caption("Tiempo con carga hidráulica y limitación de rampa del VDF.")
    else:
        st.latex(r"\boxed{t_{\mathrm{hid}} \to \infty}")
        st.warning("Con el par disponible y/o la rampa configurada, no se alcanza la velocidad objetivo (par insuficiente en el rango).")

with scol:
    st.markdown("**(B) Nota de aceleración mínima observada**")
    if reached_ok:
        st.latex(rf"\dot{{n}}_{{\min}} \approx {fmt(n_dot_min_seen)}\ \mathrm{{rpm/s}}")
    else:
        st.latex(r"\dot{n}_{\min} \le 0\ \mathrm{rpm/s}")

st.caption(
    "Modelo hidráulico empleado: $Q(n_p)=\alpha n_p$ (afinidad), "
    "$H(Q)=H_0+KQ^2$, $\\eta(Q)=\\eta_a+\\eta_b Q+\\eta_c Q^2$, "
    "$P_h=\\rho g Q H$, $P_{shaft}=P_h/\\eta(Q)$, $T_p=P_{shaft}/\\omega_p$, "
    "$T_{load,eq}=T_p/r$. La dinámica usa $J_{eq}\\,\\dot{\\omega}_m=T_{disp}-T_{load,eq}$."
)

st.divider()

# ------------------------------
# Resumen y descarga
# ------------------------------
st.markdown("### Resumen del TAG seleccionado")

sum_cols = st.columns(5)
with sum_cols[0]:
    st.latex(rf"\color{{green}}{{J_m= {fmt(J_m)}\ \mathrm{{kg\cdot m^2}}}}")
with sum_cols[1]:
    st.latex(rf"\color{{green}}{{J_{{driver}}= {fmt(J_driver)}\ \mathrm{{kg\cdot m^2}}}}")
with sum_cols[2]:
    st.latex(rf"\color{{green}}{{J_{{driven}}= {fmt(J_driven)}\ \mathrm{{kg\cdot m^2}}}}")
with sum_cols[3]:
    st.latex(rf"\color{{green}}{{J_{{imp}}= {fmt(J_imp)}\ \mathrm{{kg\cdot m^2}}}}")
with sum_cols[4]:
    st.latex(rf"\boxed{{\color{{green}}{{J_{{eq}}= {fmt(J_eq)}\ \mathrm{{kg\cdot m^2}}}}}}")

sum_cols2 = st.columns(4)
with sum_cols2[0]:
    st.latex(rf"\color{{green}}{{\Delta n= {fmt(max(n_fin_m-n_ini_m,0))}\ \mathrm{{rpm}}}}")
with sum_cols2[1]:
    st.latex(rf"\color{{green}}{{\dot{{n}}_{{torque}}= {fmt(n_dot_nohyd)}\ \mathrm{{rpm/s}}}}")
with sum_cols2[2]:
    st.latex(rf"\color{{green}}{{t_{{final,sin}}= {fmt(max(t_par,t_rampa))}\ \mathrm{{s}}}}")
with sum_cols2[3]:
    if reached_ok:
        st.latex(rf"\boxed{{\color{{green}}{{t_{{hid}}= {fmt(t_total_hyd)}\ \mathrm{{s}}}}}}")
    else:
        st.latex(rf"\boxed{{\color{{red}}{{t_{{hid}}\ \text{{no alcanzable}}}}}}")

# Tabla y descarga CSV
report = pd.DataFrame([{
    "TAG": tag_sel,
    "ratio_r": round(ratio, DEC),
    "J_m_kgm2": round(J_m, DEC),
    "J_driver_kgm2": round(J_driver, DEC),
    "J_driven_kgm2": round(J_driven, DEC),
    "J_imp_kgm2": round(J_imp, DEC),
    "J_eq_kgm2": round(J_eq, DEC),
    "n_motor_ini_rpm": round(n_ini_m, DEC),
    "n_motor_fin_rpm": round(n_fin_m, DEC),
    "Delta_n_rpm": round(max(n_fin_m-n_ini_m,0.0), DEC),
    "T_disp_Nm": round(T_disp, DEC),
    "rampa_VDF_rpms": round(rampa_vdf, DEC),
    "n_dot_torque_no_hyd_rpms": round(n_dot_nohyd, DEC),
    "t_par_s": round(t_par, DEC),
    "t_rampa_s": round(t_rampa, DEC),
    "t_final_sin_hid_s": round(t_final_nohyd, DEC),
    "H0_m": round(H0, DEC),
    "K_dataset_units": round(K, DEC),
    "eta_a": round(eta_a, DEC),
    "eta_b": round(eta_b, DEC),
    "eta_c": round(eta_c, DEC),
    "rho_kgm3": round(rho, DEC),
    "Q_min_m3h": round(Q_min, DEC),
    "Q_max_m3h": round(Q_max, DEC),
    "n_ref_rpm": round(n_ref, DEC),
    "Impeller_D_mm": round(D_imp_mm, DEC),
    "n_bomba_ini_rpm": round(n1_p, DEC),
    "n_bomba_fin_rpm": round(n2_p, DEC),
    "t_hid_s": round(t_total_hyd, DEC),
    "alcanzado": reached_ok
}])

st.dataframe(report, use_container_width=True)
st.download_button(
    "Descargar resumen (CSV)",
    data=report.to_csv(index=False).encode("utf-8"),
    file_name=f"reaccion_{tag_sel}.csv",
    mime="text/csv"
)
