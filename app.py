# -*- coding: utf-8 -*-
"""
Pumps Reaction Time – Memoria de Cálculo (Streamlit)
Requiere: bombas_dataset_with_torque_params.xlsx en la raíz.

Flujo:
1) Elegir TAG → mostrar parámetros de entrada (datos).
2) Calcular inercia equivalente reflejada al eje del motor (mostrar fórmulas).
3) Calcular tiempos de reacción SIN hidráulica (rampa VDF vs par/inercia).
4) Mostrar curva de sistema H(Q) e integrar dinámica CON carga hidráulica
   usando T_load(Q,n) = ρ g Q [H0 + K Q^2] / [ η(Q) · ω(n) ].
"""

import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# --------------------- Utilidades ---------------------

def norm(s: str) -> str:
    return (str(s).strip().lower()
            .replace(" ", "_").replace("-", "_")
            .replace("(", "").replace(")", "").replace("%", "pct"))

def pick(cols, *candidates, default=None):
    for c in candidates:
        if c in cols:
            return c
    return default

def to_float(x):
    try:
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        if s == "":
            return np.nan
        # normaliza coma decimal
        if "," in s and "." in s:
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", ".")
        return float(s)
    except Exception:
        try:
            return float(x)
        except Exception:
            return np.nan

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# --------------------- Carga de datos ---------------------

DATA_PATH = "bombas_dataset_with_torque_params.xlsx"
raw = pd.read_excel(DATA_PATH)

# mantén copia legible para mostrar
df_show = raw.copy()

# versión normalizada para el motor de cálculo
df = raw.copy()
df.columns = [norm(c) for c in df.columns]

cols = set(df.columns)

# Mapeo flexible (se adapta a nombres que ya has usado)
col_tag       = pick(cols, "tag", "eq_no", "eqno", "equipo_tag")
col_power_kw  = pick(cols, "motorpower_kw", "motorpowerefective_kw", "motorpowerinstalled_kw", "power_kw")
col_tnom_nm   = pick(cols, "t_nom_nm")
col_poles     = pick(cols, "poles")
col_r         = pick(cols, "r_trans", "relaciontransmision", "relacion_transmision", "ratio")
col_nmot_min  = pick(cols, "motor_n_min_rpm", "n_motor_min", "motorspeedmin_rpm")
col_nmot_max  = pick(cols, "motor_n_max_rpm", "n_motor_max", "motorspeedmax_rpm")
col_npum_min  = pick(cols, "pump_n_min_rpm", "n_pump_min")
col_npum_max  = pick(cols, "pump_n_max_rpm", "n_pump_max")

# Inercias (todas en kg·m² si es posible)
col_jm        = pick(cols, "motor_j_kgm2", "j_motor_kgm2")
col_jdrv      = pick(cols, "j_driver_total_kgm2", "driver_total_j_kgm2")
col_jdrn      = pick(cols, "j_driven_total_kgm2", "driven_total_j_kgm2")
col_jimp      = pick(cols, "impeller_j_kgm2", "j_impeller_kgm2", "j_impulsor_kgm2")
col_dimp_mm   = pick(cols, "impeller_d_mm", "diametroimpulsor_mm")

# Parámetros hidráulicos
col_H0        = pick(cols, "h0_m")
col_K         = pick(cols, "k_m_s2")
col_eta_a     = pick(cols, "eta_a")
col_eta_b     = pick(cols, "eta_b")
col_eta_c     = pick(cols, "eta_c")
col_rho       = pick(cols, "rho_kgm3")
col_nref      = pick(cols, "n_ref_rpm")
col_Qmin_h    = pick(cols, "q_min_m3h")
col_Qmax_h    = pick(cols, "q_max_m3h")

# Convierte posibles columnas numéricas
for c in [col_power_kw, col_tnom_nm, col_poles, col_r,
          col_nmot_min, col_nmot_max, col_npum_min, col_npum_max,
          col_jm, col_jdrv, col_jdrn, col_jimp, col_dimp_mm,
          col_H0, col_K, col_eta_a, col_eta_b, col_eta_c, col_rho, col_nref, col_Qmin_h, col_Qmax_h]:
    if c and (df[c].dtype == object):
        df[c] = df[c].apply(to_float)

# --------------------- UI ---------------------

st.set_page_config(page_title="Tiempo de reacción de bombas", layout="wide")
st.title("Memoria de Cálculo – Tiempo de reacción de bombas (VDF)")

st.sidebar.header("Selección")
tags = sorted(df[col_tag].dropna().astype(str).unique().tolist())
tag = st.sidebar.selectbox("TAG", tags)

# Parámetros globales de simulación
st.sidebar.header("Parámetros de simulación")
ramp_rpm_s = st.sidebar.number_input("Rampa VDF [rpm/s] (eje motor)", min_value=1.0, value=300.0, step=10.0)
dt = st.sidebar.number_input("Paso de integración dt [s] (hidráulica)", min_value=0.001, value=0.01, step=0.01, format="%.3f")

row = df[df[col_tag].astype(str) == str(tag)].iloc[0]

# --------------------- 1) Mostrar datos de entrada ---------------------

st.header("1) Parámetros de entrada (datos)")
colA, colB, colC = st.columns(3)

with colA:
    st.subheader("Motor / Transmisión")
    st.write(f"**Potencia motor:** {row.get(col_power_kw, np.nan)} kW")
    st.write(f"**Polos:** {int(row[col_poles]) if not pd.isna(row.get(col_poles)) else '—'}")
    st.write(f"**Relación transmisión (r = ω_motor/ω_bomba):** {row.get(col_r, np.nan)}")
    st.write(f"**Velocidad motor min–max [rpm]:** {row.get(col_nmot_min, np.nan)} – {row.get(col_nmot_max, np.nan)}")
    if col_npum_min and col_npum_max in df.columns:
        st.write(f"**Velocidad bomba min–max [rpm]:** {row.get(col_npum_min, np.nan)} – {row.get(col_npum_max, np.nan)}")

with colB:
    st.subheader("Inercias por elemento (kg·m²)")
    st.write(f"**Motor (J_m):** {row.get(col_jm, np.nan)}")
    st.write(f"**Polea + manguito conductor (J_driver):** {row.get(col_jdrv, np.nan)}")
    st.write(f"**Polea + manguito conducido (J_driven):** {row.get(col_jdrn, np.nan)}")
    st.write(f"**Impulsor (J_imp):** {row.get(col_jimp, np.nan)}")

with colC:
    st.subheader("Impulsor / Hidráulica (datos)")
    if col_dimp_mm:
        st.write(f"**Diámetro impulsor:** {row.get(col_dimp_mm, np.nan)} mm")
    st.write(f"**H0 (m):** {row.get(col_H0, np.nan)}")
    st.write(f"**K (m·s⁻²):** {row.get(col_K, np.nan)}")
    st.write(f"**η(Q) = a + bQ + cQ²** con **Q en m³/s**")
    st.write(f"a = {row.get(col_eta_a, np.nan)}, b = {row.get(col_eta_b, np.nan)}, c = {row.get(col_eta_c, np.nan)}")
    st.write(f"**ρ (kg/m³):** {row.get(col_rho, np.nan)}")
    st.write(f"**n_ref (rpm):** {row.get(col_nref, np.nan)}")
    st.write(f"**Q rango (m³/h):** {row.get(col_Qmin_h, np.nan)} – {row.get(col_Qmax_h, np.nan)}")

# --------------------- 2) Inercia equivalente al eje del motor ---------------------

st.header("2) Inercia equivalente reflejada al eje del motor")

st.markdown(r"""
**Fórmula:**
\[
J_{\text{eq}} \;=\; J_m \;+\; J_{\text{driver}} \;+\; \frac{J_{\text{driven}} + J_{\text{imp}}}{r^2}
\]
donde \(r = \omega_{\text{motor}} / \omega_{\text{bomba}}\).
""")

J_m   = float(row.get(col_jm, 0.0) or 0.0)
J_drv = float(row.get(col_jdrv, 0.0) or 0.0)
J_drn = float(row.get(col_jdrn, 0.0) or 0.0)
J_imp = float(row.get(col_jimp, 0.0) or 0.0)
r_tr  = float(row.get(col_r, np.nan) or np.nan)

J_eq = np.nan
if not (pd.isna(r_tr) or r_tr == 0):
    J_eq = J_m + J_drv + (J_drn + J_imp)/(r_tr**2)

st.info(f"**J_eq (kg·m²):** {np.round(J_eq, 6)}")

# --------------------- 3) Tiempo de reacción SIN hidráulica ---------------------

st.header("3) Tiempo de reacción **sin** considerar la carga hidráulica")

st.markdown(r"""
**Par disponible:**  
- Si existe en el dataset: \(T_{\text{nom}}\) (columna).  
- Si no: \(\displaystyle T_{\text{nom}} = \frac{9550 \, P_{\text{motor}}[\text{kW}]}{n_{\text{motor,max}}[\text{rpm}]}\).

**Aceleración por par (eje motor):**
\[
\dot n_{\text{torque}} \;=\; \frac{60}{2\pi}\,\frac{T_{\text{nom}}}{J_{\text{eq}}}\quad[\text{rpm/s}]
\]

**Tiempo por par/inercia:**
\[
t_{\text{par}} \;=\; \frac{\Delta n_{\text{motor}}}{\dot n_{\text{torque}}}
\]

**Tiempo por rampa del VDF:**
\[
t_{\text{rampa}} \;=\; \frac{\Delta n_{\text{motor}}}{\text{rampa}_{\text{VDF}}}
\]

**Tiempo final (sin hidráulica):**
\[
t_{\text{final,sin}} \;=\; \max\left(t_{\text{par}},\,t_{\text{rampa}}\right)
\]
""")

nmin = float(row.get(col_nmot_min, 0.0) or 0.0)
nmax = float(row.get(col_nmot_max, 0.0) or 0.0)

c1, c2, c3 = st.columns(3)
with c1:
    n_ini = st.number_input("n_motor inicial [rpm]", value=float(nmin), step=10.0)
with c2:
    n_fin = st.number_input("n_motor final [rpm]",  value=float(nmax), step=10.0)
with c3:
    # Par disponible: columna o cálculo
    if col_tnom_nm and not pd.isna(row.get(col_tnom_nm)):
        T_nom = float(row[col_tnom_nm])
    else:
        P_kw = float(row.get(col_power_kw, 0.0) or 0.0)
        n_for_t = float(row.get(col_nmot_max, n_fin) or n_fin or 1.0)
        T_nom = 9550.0 * P_kw / max(n_for_t, 1.0)
    st.number_input("Par motor disponible T_nom [Nm]", value=float(T_nom), step=10.0, key="tnom_view")

delta_n = max(0.0, n_fin - n_ini)
accel_torque = (60.0/(2.0*math.pi)) * (T_nom / max(J_eq, 1e-9))  # rpm/s
t_par   = delta_n / max(accel_torque, 1e-9)
t_rampa = delta_n / max(ramp_rpm_s, 1e-9)
t_final_nohyd = max(t_par, t_rampa)

st.success(f"**Δn_motor:** {delta_n:.1f} rpm  |  **ẋ_n (por par):** {accel_torque:.1f} rpm/s")
st.info(f"**t_par:** {t_par:.2f} s  |  **t_rampa:** {t_rampa:.2f} s  →  **t_final (sin hidráulica): {t_final_nohyd:.2f} s**")

# --------------------- 4) Curva de sistema y tiempo con carga hidráulica ---------------------

st.header("4) Curva de sistema y tiempo de reacción **con** carga hidráulica")

H0   = float(row.get(col_H0, np.nan) or np.nan)
K    = float(row.get(col_K,  np.nan) or np.nan)
eta_a = float(row.get(col_eta_a, 0.0) or 0.0)
eta_b = float(row.get(col_eta_b, 0.0) or 0.0)
eta_c = float(row.get(col_eta_c, 0.0) or 0.0)
rho   = float(row.get(col_rho, 1000.0) or 1000.0)

# Modelo de eficiencia y curva de sistema
def eta_of_Q(Q_m3s):
    return eta_a + eta_b*Q_m3s + eta_c*(Q_m3s**2)

def H_of_Q(Q_m3s):
    return H0 + K*(Q_m3s**2)

# Q(n): por afinidad lineal (Q ≈ α * n_bomba). α por defecto desde datos.
n_ref = float(row.get(col_nref, n_fin) or n_fin or 1.0)
Qmin_h = float(row.get(col_Qmin_h, 0.0) or 0.0)
Qmax_h = float(row.get(col_Qmax_h, 0.0) or 0.0)
Qref_h = (Qmin_h + Qmax_h)/2.0 if (Qmax_h > 0) else 0.0
alpha_default = (Qref_h / max(n_ref, 1.0))  # m3/h por rpm
alpha_user = st.number_input("Coeficiente de caudal por rpm de bomba (α) [m³/h por rpm]", value=float(alpha_default), step=1.0)
alpha = alpha_user / 3600.0  # → m3/s por rpm

st.caption("Se usa **Q(n_bomba)=α·n_bomba** como aproximación de afinidad. Puedes ajustar α si tienes un punto de referencia específico.")

# Gráfica H(Q)
Qh = np.linspace(max(Qmin_h*0.5, 1.0), max(Qmax_h*1.2, Qref_h*1.5, 50.0), 100)
Qs = Qh/3600.0
Hvals = H_of_Q(Qs)

fig = go.Figure()
fig.add_trace(go.Scatter(x=Qh, y=Hvals, mode="lines", name="Sistema H(Q)=H0+KQ²",
                         line=dict(width=3)))
fig.update_layout(xaxis_title="Q [m³/h]", yaxis_title="H [m]",
                  template="plotly_white", height=420)
st.plotly_chart(fig, use_container_width=True)

st.markdown(r"""
**Par resistente (eje bomba):**
\[
T_{\text{pump}}(Q,n) \;=\; \frac{\rho\,g\,Q\,[\,H_0 + KQ^2\,]}{\eta(Q)\,\omega(n)}\,,
\quad \omega(n)=2\pi n/60
\]
**Par en el eje motor:** \(T_{\text{motor}} = T_{\text{pump}}/r\).

**Ecuación dinámica (eje motor, integración explícita):**
\[
\frac{dn_{\text{motor}}}{dt} \;=\;
\min\!\left(\text{rampa}_{\text{VDF}},\;\frac{60}{2\pi}\,\frac{T_{\text{nom}}-T_{\text{motor}}(n)}{J_{\text{eq}}}\right).
\]
""")

# Integración con carga hidráulica
def simulate_with_hyd(n0, n1, dt, T_nom, J_eq, r_tr, alpha_m3s_per_rpm, H0, K, eta_a, eta_b, eta_c, rho, ramp_rpm_s, nmax_iter=2_000_000):
    t = 0.0
    n = n0
    times = [0.0]
    n_list = [n0]
    q_list = [0.0]
    torque_load_motor = [0.0]
    g = 9.80665
    while n < n1 and len(times) < nmax_iter:
        # Setpoint por rampa VDF
        n_cmd = min(n1, n + ramp_rpm_s*dt)

        # Variables bomba
        n_bomba = n / max(r_tr, 1e-9)
        Q = max(0.0, alpha_m3s_per_rpm * n_bomba)   # m3/s
        eta_q = clamp(eta_of_Q(Q), 0.3, 0.9)
        H_q = H_of_Q(Q)
        omega_b = 2.0*math.pi*max(n_bomba, 1e-6)/60.0

        # Par de carga → eje motor
        P_h = rho * g * Q * H_q                 # W
        T_pump = (P_h / max(eta_q, 1e-6)) / omega_b   # Nm (eje bomba)
        T_motor_load = T_pump / max(r_tr, 1e-9)       # Nm (eje motor)

        # Aceleración por par disponible
        accel_torque = (60.0/(2.0*math.pi)) * ((T_nom - T_motor_load) / max(J_eq, 1e-9))  # rpm/s
        accel = min(ramp_rpm_s, accel_torque)
        accel = max(accel, 0.0)  # no frenamos en esta simulación de subida

        n_next = n + accel*dt
        n_next = min(n_next, n_cmd)  # no excede rampa VDF

        t += dt
        n = n_next
        times.append(t)
        n_list.append(n)
        q_list.append(Q*3600.0)  # m3/h
        torque_load_motor.append(T_motor_load)

        if n >= n1 - 1e-6:
            break

    return pd.DataFrame({
        "t_s": times,
        "n_motor_rpm": n_list,
        "Q_m3h": q_list,
        "T_load_motor_Nm": torque_load_motor
    })

if not any(pd.isna([J_eq, r_tr, H0, K, rho])):
    sim = simulate_with_hyd(
        n0=n_ini, n1=n_fin, dt=dt, T_nom=T_nom, J_eq=J_eq, r_tr=r_tr,
        alpha_m3s_per_rpm=alpha, H0=H0, K=K,
        eta_a=eta_a, eta_b=eta_b, eta_c=eta_c, rho=rho,
        ramp_rpm_s=ramp_rpm_s
    )

    t_hyd = float(sim["t_s"].iloc[-1]) if len(sim) else np.nan

    # Gráficas n(t) y Q(t)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=sim["t_s"], y=sim["n_motor_rpm"], mode="lines", name="n_motor(t)"))
    fig2.update_layout(template="plotly_white", xaxis_title="t [s]", yaxis_title="n_motor [rpm]", height=380)
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=sim["t_s"], y=sim["Q_m3h"], mode="lines", name="Q(t)"))
    fig3.update_layout(template="plotly_white", xaxis_title="t [s]", yaxis_title="Q [m³/h]", height=360)
    st.plotly_chart(fig3, use_container_width=True)

    st.success(f"**Tiempo de reacción con hidráulica: {t_hyd:.2f} s**")
else:
    st.warning("Faltan parámetros para la simulación hidráulica (H0, K, ρ, r o J_eq).")

# --------------------- Exportar resultados ---------------------

st.header("5) Exportar resultados")
summary = {
    "TAG": tag,
    "J_m_kgm2": J_m, "J_driver_kgm2": J_drv, "J_driven_kgm2": J_drn, "J_imp_kgm2": J_imp,
    "r_trans": r_tr, "J_eq_kgm2": J_eq,
    "T_nom_Nm": T_nom, "rampa_rpm_s": ramp_rpm_s,
    "n_ini_rpm": n_ini, "n_fin_rpm": n_fin, "delta_n_rpm": delta_n,
    "accel_rpm_s_torque": accel_torque, "t_par_s": t_par, "t_rampa_s": t_rampa,
    "t_final_sin_hidraulica_s": t_final_nohyd,
    "H0_m": H0, "K_m_s2": K, "rho_kgm3": rho,
    "eta_a": eta_a, "eta_b": eta_b, "eta_c": eta_c,
}
summary_df = pd.DataFrame([summary])
st.dataframe(summary_df, use_container_width=True)

csv_bytes = summary_df.to_csv(index=False).encode("utf-8")
st.download_button("Descargar resumen (CSV)", data=csv_bytes, file_name=f"resumen_{tag}.csv", mime="text/csv")

st.caption("Nota: el modelo hidráulico usa Q(n_bomba)=α·n_bomba por afinidad. Ajusta α si dispones de un punto de referencia específico.")
