# -*- coding: utf-8 -*-
"""
Memoria de Cálculo – Tiempo de reacción de bombas (VDF)
Poner en la raíz: bombas_dataset_with_torque_params.xlsx

Suposiciones:
- La PRIMERA columna del Excel contiene los TAG (únicos).
- Las inercias de polea y manguito están en SI (kg·m²).
"""

import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ------------------ utilidades ------------------
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
        if pd.isna(x): return np.nan
        s = str(x).strip()
        if s == "": return np.nan
        if "," in s and "." in s:
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", ".")
        return float(s)
    except Exception:
        try: return float(x)
        except Exception: return np.nan

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

# ------------------ carga de datos ------------------
DATA_PATH = "bombas_dataset_with_torque_params.xlsx"
raw = pd.read_excel(DATA_PATH)
df = raw.copy()
df.columns = [norm(c) for c in df.columns]
cols = set(df.columns)

# Genera columnas de inercias totales en SI (si no existen)
for c in [
    "driverpulley_j_kgm2","driverbushing_j_kgm2",
    "drivenpulley_j_kgm2","drivenbushing_j_kgm2",
]:
    if c in df.columns and df[c].dtype == object:
        df[c] = df[c].apply(to_float)

if "j_driver_total_kgm2" not in df.columns:
    df["j_driver_total_kgm2"] = \
        (df["driverpulley_j_kgm2"] if "driverpulley_j_kgm2" in df.columns else 0.0) + \
        (df["driverbushing_j_kgm2"] if "driverbushing_j_kgm2" in df.columns else 0.0)

if "j_driven_total_kgm2" not in df.columns:
    df["j_driven_total_kgm2"] = \
        (df["drivenpulley_j_kgm2"] if "drivenpulley_j_kgm2" in df.columns else 0.0) + \
        (df["drivenbushing_j_kgm2"] if "drivenbushing_j_kgm2" in df.columns else 0.0)

# Mapeo flexible de columnas
col_tag      = df.columns[0]  # primera columna = TAG
col_power_kw = pick(cols, "motorpower_kw","motorpowerefective_kw","motorpowerinstalled_kw","power_kw")
col_poles    = pick(cols, "poles")
col_r        = pick(cols, "r_trans","relaciontransmision","relacion_transmision")
col_nmot_min = pick(cols, "motorspeedmin_rpm","motor_n_min_rpm","n_motor_min")
col_nmot_max = pick(cols, "motorspeedmax_rpm","motor_n_max_rpm","n_motor_max")
col_npum_min = pick(cols, "pump_n_min_rpm","n_pump_min")
col_npum_max = pick(cols, "pump_n_max_rpm","n_pump_max")

# Inercias
col_jm   = pick(cols, "motor_j_kgm2","j_motor_kgm2")
col_jdrv = pick(cols, "j_driver_total_kgm2")
col_jdrn = pick(cols, "j_driven_total_kgm2")
col_jimp = pick(cols, "impeller_j_kgm2","j_impeller_kgm2","j_impulsor_kgm2")
col_dimp = pick(cols, "impeller_d_mm","diametroimpulsor_mm")
col_tnom = pick(cols, "t_nom_nm")  # opcional

# Hidráulica
col_H0     = pick(cols, "h0_m")
col_K      = pick(cols, "k_m_s2")
col_eta_a  = pick(cols, "eta_a")
col_eta_b  = pick(cols, "eta_b")
col_eta_c  = pick(cols, "eta_c")
col_rho    = pick(cols, "rho_kgm3")
col_nref   = pick(cols, "n_ref_rpm")
col_Qmin_h = pick(cols, "q_min_m3h")
col_Qmax_h = pick(cols, "q_max_m3h")

# Convierte a numérico lo que falte
for c in [col_power_kw, col_poles, col_r, col_nmot_min, col_nmot_max, col_npum_min, col_npum_max,
          col_jm, col_jdrv, col_jdrn, col_jimp, col_dimp, col_tnom,
          col_H0, col_K, col_eta_a, col_eta_b, col_eta_c, col_rho, col_nref, col_Qmin_h, col_Qmax_h]:
    if c and df[c].dtype == object:
        df[c] = df[c].apply(to_float)

# ------------------ UI ------------------
st.set_page_config(page_title="Memoria de Cálculo – Bombas (VDF)", layout="wide")
st.title("Memoria de Cálculo – Tiempo de reacción de bombas (VDF)")

# Selección de TAG por posición (primera columna)
tag_series = df.iloc[:, 0].astype(str)
tags = sorted(tag_series.unique().tolist())
tag = st.sidebar.selectbox("TAG", tags)

st.sidebar.header("Parámetros de simulación")
ramp_rpm_s = st.sidebar.number_input("Rampa VDF [rpm/s] (eje motor)", min_value=1.0, value=300.0, step=10.0)
dt = st.sidebar.number_input("Paso de integración dt [s] (hidráulica)", min_value=0.001, value=0.01, step=0.01, format="%.3f")

row = df.loc[tag_series == tag].iloc[0]

# ------------------ 1) Entradas ------------------
st.header("1) Parámetros de entrada (dato)")
c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("Motor / Transmisión")
    st.write(f"**Potencia motor (kW):** {row.get(col_power_kw, np.nan)}")
    if col_poles:
        st.write(f"**Polos:** {int(row[col_poles]) if not pd.isna(row.get(col_poles)) else '—'}")
    st.write(f"**Relación r = ω_motor/ω_bomba:** {row.get(col_r, np.nan)}")
    st.write(f"**n_motor min–max [rpm]:** {row.get(col_nmot_min, np.nan)} – {row.get(col_nmot_max, np.nan)}")
    if (col_npum_min in df.columns) and (col_npum_max in df.columns):
        st.write(f"**n_bomba min–max [rpm]:** {row.get(col_npum_min, np.nan)} – {row.get(col_npum_max, np.nan)}")

with c2:
    st.subheader("Inercias (kg·m²)")
    st.write(f"**J_m (motor):** {row.get(col_jm, np.nan)}")
    # Desglose conductor
    drv_p = float(row.get("driverpulley_j_kgm2", np.nan)) if "driverpulley_j_kgm2" in df.columns else np.nan
    drv_b = float(row.get("driverbushing_j_kgm2", np.nan)) if "driverbushing_j_kgm2" in df.columns else np.nan
    st.write(f"**J_driver (total):** {row.get(col_jdrv, np.nan)}  \n  └ = polea({drv_p}) + manguito({drv_b})")
    # Desglose conducido
    drn_p = float(row.get("drivenpulley_j_kgm2", np.nan)) if "drivenpulley_j_kgm2" in df.columns else np.nan
    drn_b = float(row.get("drivenbushing_j_kgm2", np.nan)) if "drivenbushing_j_kgm2" in df.columns else np.nan
    st.write(f"**J_driven (total):** {row.get(col_jdrn, np.nan)}  \n  └ = polea({drn_p}) + manguito({drn_b})")
    st.write(f"**J_imp (impulsor):** {row.get(col_jimp, np.nan)}")

with c3:
    st.subheader("Impulsor / Hidráulica")
    if col_dimp: st.write(f"**D_imp (mm):** {row.get(col_dimp, np.nan)}")
    st.write(f"**H0 (m):** {row.get(col_H0, np.nan)}")
    st.write(f"**K (m·s⁻²):** {row.get(col_K, np.nan)}")
    st.write("**η(Q)=a+bQ+cQ²** (Q en m³/s)")
    st.write(f"a={row.get(col_eta_a, np.nan)}, b={row.get(col_eta_b, np.nan)}, c={row.get(col_eta_c, np.nan)}")
    st.write(f"**ρ (kg/m³):** {row.get(col_rho, np.nan)}")
    st.write(f"**n_ref (rpm):** {row.get(col_nref, np.nan)}")
    st.write(f"**Q rango (m³/h):** {row.get(col_Qmin_h, np.nan)} – {row.get(col_Qmax_h, np.nan)}")

# ------------------ 2) Inercia equivalente ------------------
st.header("2) Inercia equivalente al eje del motor")
st.latex(r"J_{\mathrm{eq}} = J_m + J_{\mathrm{driver}} + \frac{J_{\mathrm{driven}} + J_{\mathrm{imp}}}{r^2}")

J_m   = float(row.get(col_jm, 0.0) or 0.0)
J_drv = float(row.get(col_jdrv, 0.0) or 0.0)
J_drn = float(row.get(col_jdrn, 0.0) or 0.0)
J_imp = float(row.get(col_jimp, 0.0) or 0.0)
r_tr  = float(row.get(col_r, np.nan) or np.nan)

J_eq = np.nan if (pd.isna(r_tr) or r_tr == 0) else J_m + J_drv + (J_drn + J_imp)/(r_tr**2)
st.info(f"**J_eq (kg·m²):** {np.round(J_eq, 6)}")

# ------------------ 3) Tiempo de reacción sin hidráulica ------------------
st.header("3) Tiempo de reacción **sin** hidráulica")
st.latex(r"\dot n_{\mathrm{torque}} = \frac{60}{2\pi}\,\frac{T_{\mathrm{nom}}}{J_{\mathrm{eq}}}")
st.latex(r"t_{\mathrm{par}}=\frac{\Delta n}{\dot n_{\mathrm{torque}}},\quad "
         r"t_{\mathrm{rampa}}=\frac{\Delta n}{\mathrm{rampa}_{\mathrm{VDF}}},\quad "
         r"t_{\mathrm{final,sin}}=\max(t_{\mathrm{par}},\,t_{\mathrm{rampa}})")

# Usar directamente los valores del dataset (sin inputs editables)
nmin = float(row.get(col_nmot_min, 0.0) or 0.0)
nmax = float(row.get(col_nmot_max, 0.0) or 0.0)
# Par nominal: si no está, se estima con P y n_max
if col_tnom and not pd.isna(row.get(col_tnom)):
    T_nom = float(row[col_tnom])
else:
    P_kw = float(row.get(col_power_kw, 0.0) or 0.0)
    n_for_t = float(row.get(col_nmot_max, nmax) or nmax or 1.0)
    T_nom = 9550.0 * P_kw / max(n_for_t, 1.0)

st.write(f"**n_motor inicial [rpm]:** {nmin}")
st.write(f"**n_motor final [rpm]:** {nmax}")
st.write(f"**Par disponible T_nom [Nm]:** {np.round(T_nom,2)}")

delta_n = max(0.0, nmax - nmin)
accel_torque = (60.0/(2.0*math.pi)) * (T_nom / max(J_eq, 1e-9))  # rpm/s
t_par   = delta_n / max(accel_torque, 1e-9)
t_rampa = delta_n / max(ramp_rpm_s, 1e-9)
t_nohyd = max(t_par, t_rampa)
st.success(f"Δn={delta_n:.1f} rpm | ẋ_n={accel_torque:.1f} rpm/s  →  t_par={t_par:.2f} s, "
           f"t_rampa={t_rampa:.2f} s, **t_final(sin)={t_nohyd:.2f} s**")

# ------------------ 4) Curva de sistema + hidráulica ------------------
st.header("4) Curva de sistema y tiempo **con** hidráulica")
st.latex(r"H(Q)=H_0+K\,Q^2")
st.latex(r"T_{\mathrm{pump}}(Q,n)=\dfrac{\rho\,g\,Q\,[H_0+KQ^2]}{\eta(Q)\,\omega(n)},\quad "
         r"\omega(n)=\dfrac{2\pi n}{60},\qquad T_{\mathrm{motor}}=\dfrac{T_{\mathrm{pump}}}{r}")
st.latex(r"\dfrac{dn_{\mathrm{motor}}}{dt}=\min\!\Big(\mathrm{rampa}_{\mathrm{VDF}},\; "
         r"\dfrac{60}{2\pi}\dfrac{T_{\mathrm{nom}}-T_{\mathrm{motor}}(n)}{J_{\mathrm{eq}}}\Big)")

H0     = float(row.get(col_H0, np.nan) or np.nan)
K      = float(row.get(col_K,  np.nan) or np.nan)
eta_a  = float(row.get(col_eta_a, 0.0) or 0.0)
eta_b  = float(row.get(col_eta_b, 0.0) or 0.0)
eta_c  = float(row.get(col_eta_c, 0.0) or 0.0)
rho    = float(row.get(col_rho, 1000.0) or 1000.0)

def eta_of_Q(Q_m3s): return eta_a + eta_b*Q_m3s + eta_c*(Q_m3s**2)
def H_of_Q(Q_m3s):   return H0 + K*(Q_m3s**2)

n_ref  = float(row.get(col_nref, nmax) or nmax or 1.0)
Qmin_h = float(row.get(col_Qmin_h, 0.0) or 0.0)
Qmax_h = float(row.get(col_Qmax_h, 0.0) or 0.0)
Qref_h = (Qmin_h + Qmax_h)/2.0 if (Qmax_h and Qmax_h > 0) else 0.0

alpha_default = (Qref_h / max(n_ref, 1.0))  # m³/h por rpm de bomba
alpha_user = st.number_input("α: caudal por rpm de bomba [m³/h·rpm]", value=float(alpha_default), step=1.0)
alpha = alpha_user / 3600.0                  # → m³/s por rpm

# H(Q)
Qh = np.linspace(max(Qmin_h*0.5, 1.0), max(Qmax_h*1.2, max(50.0, Qref_h*1.5)), 120)
figH = go.Figure()
figH.add_trace(go.Scatter(x=Qh, y=H_of_Q(Qh/3600.0), mode="lines", name="H(Q)=H0+KQ²", line=dict(width=3)))
figH.update_layout(xaxis_title="Q [m³/h]", yaxis_title="H [m]", template="plotly_white", height=400)
st.plotly_chart(figH, use_container_width=True)

def simulate_with_hyd(n0, n1, dt, T_nom, J_eq, r_tr, alpha_m3s_per_rpm,
                      H0, K, eta_a, eta_b, eta_c, rho, ramp_rpm_s):
    g = 9.80665
    t, n = 0.0, n0
    times, n_list, q_list = [0.0], [n0], [0.0]
    max_steps = int(1e6)
    steps = 0
    while n < n1 and steps < max_steps:
        steps += 1
        n_cmd = min(n1, n + ramp_rpm_s*dt)
        n_b   = n / max(r_tr, 1e-9)
        Q     = max(0.0, alpha_m3s_per_rpm * n_b)   # m3/s
        eta   = clamp(eta_of_Q(Q), 0.3, 0.9)
        H     = H_of_Q(Q)
        omega_b = 2*math.pi*max(n_b,1e-6)/60
        P_h   = rho * g * Q * H
        T_pump = (P_h / max(eta, 1e-6)) / omega_b
        T_mot_load = T_pump / max(r_tr, 1e-9)

        accel_torque = (60/(2*math.pi)) * ((T_nom - T_mot_load)/max(J_eq,1e-9))
        accel = min(ramp_rpm_s, max(accel_torque, 0.0))
        n = min(n_cmd, n + accel*dt)
        t += dt

        times.append(t); n_list.append(n); q_list.append(Q*3600.0)
        if n >= n1 - 1e-6: break

    return pd.DataFrame({"t_s": times, "n_motor_rpm": n_list, "Q_m3h": q_list})

if not any(pd.isna([J_eq, r_tr, H0, K, rho])):
    sim = simulate_with_hyd(
        n0=nmin, n1=nmax, dt=dt, T_nom=T_nom, J_eq=J_eq, r_tr=r_tr,
        alpha_m3s_per_rpm=alpha, H0=H0, K=K,
        eta_a=eta_a, eta_b=eta_b, eta_c=eta_c, rho=rho, ramp_rpm_s=ramp_rpm_s
    )
    t_hyd = float(sim["t_s"].iloc[-1])

    figN = go.Figure()
    figN.add_trace(go.Scatter(x=sim["t_s"], y=sim["n_motor_rpm"], mode="lines", name="n_motor(t)"))
    figN.update_layout(template="plotly_white", xaxis_title="t [s]", yaxis_title="n_motor [rpm]", height=360)
    st.plotly_chart(figN, use_container_width=True)

    figQ = go.Figure()
    figQ.add_trace(go.Scatter(x=sim["t_s"], y=sim["Q_m3h"], mode="lines", name="Q(t)"))
    figQ.update_layout(template="plotly_white", xaxis_title="t [s]", yaxis_title="Q [m³/h]", height=340)
    st.plotly_chart(figQ, use_container_width=True)

    st.success(f"**Tiempo de reacción con hidráulica: {t_hyd:.2f} s**")
else:
    st.warning("Faltan parámetros para la simulación hidráulica (H0, K, ρ, r o J_eq).")

# ------------------ 5) Exportar ------------------
st.header("5) Exportar resumen")
summary = pd.DataFrame([{
    "TAG": str(tag),
    "J_eq_kgm2": J_eq, "T_nom_Nm": T_nom, "r_trans": r_tr,
    "n_ini_rpm": nmin, "n_fin_rpm": nmax, "delta_n_rpm": (nmax - nmin),
    "accel_rpm_s_torque": accel_torque, "t_par_s": t_par,
    "t_rampa_s": t_rampa, "t_final_sin_hidraulica_s": t_nohyd,
    "H0_m": H0, "K_m_s2": K, "rho_kgm3": rho,
    "eta_a": eta_a, "eta_b": eta_b, "eta_c": eta_c
}])
st.dataframe(summary, use_container_width=True)
st.download_button(
    "Descargar resumen (CSV)",
    data=summary.to_csv(index=False).encode("utf-8"),
    file_name=f"resumen_{tag}.csv",
    mime="text/csv"
)
