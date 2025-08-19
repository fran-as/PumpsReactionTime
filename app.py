# -*- coding: utf-8 -*-
"""
Pumps Reaction Time – Memoria de Cálculo (Streamlit)
Archivo esperado en la raíz: bombas_dataset_with_torque_params.xlsx
"""

import re
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ---------------- Utilidades ----------------
def norm(s: str) -> str:
    return (str(s).strip().lower()
            .replace(" ", "_").replace("-", "_")
            .replace("(", "").replace(")", "").replace("%", "pct"))

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

def clamp(x, lo, hi): return max(lo, min(hi, x))

def series_from_any(df: pd.DataFrame, col):
    """Devuelve una Series 'segura' aunque haya columnas duplicadas o objetos raros."""
    obj = df[col]
    if isinstance(obj, pd.Series):
        return obj
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0]
    # última barrera
    return pd.Series(obj)

# ---------------- Carga de datos ----------------
DATA_PATH = "bombas_dataset_with_torque_params.xlsx"
raw = pd.read_excel(DATA_PATH)   # copia para mostrar si quieres
df = raw.copy()
df.columns = [norm(c) for c in df.columns]
cols = list(df.columns)
cols_set = set(cols)

st.set_page_config(page_title="Tiempo de reacción de bombas (VDF)", layout="wide")
st.title("Memoria de Cálculo – Tiempo de reacción de bombas (VDF)")

st.sidebar.header("Selección de columnas")

# --------- Detección robusta de la columna TAG ----------
def looks_like_tag_col(df: pd.DataFrame, colname: str) -> bool:
    s = series_from_any(df, colname).astype(str).head(250).tolist()
    return any(re.search(r"\b\d{4}-PU-\d{3}\b", v) for v in s)

# 1) por nombre + regex
name_hint = [c for c in cols if any(k in c for k in ["tag", "eq_no", "eqno", "equipo"])]
detected = None
for c in name_hint:
    if looks_like_tag_col(df, c):
        detected = c
        break

# 2) si no, barrido de todas las columnas por regex en valores
if detected is None:
    for c in cols:
        try:
            if looks_like_tag_col(df, c):
                detected = c
                break
        except Exception:
            continue

# 3) selección manual si aún no se detecta
if detected is None:
    st.sidebar.warning("No se detectó automáticamente la columna de TAG. Selecciónala manualmente.")
    detected = st.sidebar.selectbox("Columna TAG", options=cols)
else:
    st.sidebar.caption(f"Columna TAG detectada: **{detected}** (puedes cambiarla abajo)")
    detected = st.sidebar.selectbox("Columna TAG", options=cols, index=cols.index(detected))

col_tag = detected

# --------- Mapeo flexible del resto de columnas ----------
def pick(*candidates, default=None):
    for c in candidates:
        if c in cols_set: return c
    return default

col_power_kw  = pick("motorpower_kw", "motorpowerefective_kw", "motorpowerinstalled_kw", "power_kw")
col_tnom_nm   = pick("t_nom_nm")
col_poles     = pick("poles")
col_r         = pick("r_trans", "relaciontransmision", "relacion_transmision", "ratio")
col_nmot_min  = pick("motor_n_min_rpm", "n_motor_min", "motorspeedmin_rpm")
col_nmot_max  = pick("motor_n_max_rpm", "n_motor_max", "motorspeedmax_rpm")
col_npum_min  = pick("pump_n_min_rpm", "n_pump_min")
col_npum_max  = pick("pump_n_max_rpm", "n_pump_max")

# Inercias
col_jm        = pick("motor_j_kgm2", "j_motor_kgm2")
col_jdrv      = pick("j_driver_total_kgm2", "driver_total_j_kgm2")
col_jdrn      = pick("j_driven_total_kgm2", "driven_total_j_kgm2")
col_jimp      = pick("impeller_j_kgm2", "j_impeller_kgm2", "j_impulsor_kgm2")
col_dimp_mm   = pick("impeller_d_mm", "diametroimpulsor_mm")

# Hidráulica
col_H0        = pick("h0_m")
col_K         = pick("k_m_s2")
col_eta_a     = pick("eta_a")
col_eta_b     = pick("eta_b")
col_eta_c     = pick("eta_c")
col_rho       = pick("rho_kgm3")
col_nref      = pick("n_ref_rpm")
col_Qmin_h    = pick("q_min_m3h")
col_Qmax_h    = pick("q_max_m3h")

# tipado numérico
for c in [col_power_kw, col_tnom_nm, col_poles, col_r,
          col_nmot_min, col_nmot_max, col_npum_min, col_npum_max,
          col_jm, col_jdrv, col_jdrn, col_jimp, col_dimp_mm,
          col_H0, col_K, col_eta_a, col_eta_b, col_eta_c, col_rho, col_nref, col_Qmin_h, col_Qmax_h]:
    if c and (df[c].dtype == object):
        df[c] = df[c].apply(to_float)

# ---------------- UI – selección TAG ----------------
tag_series = series_from_any(df, col_tag).astype(str)
tags = sorted(tag_series.unique().tolist())
tag = st.sidebar.selectbox("TAG", tags)

st.sidebar.header("Parámetros de simulación")
ramp_rpm_s = st.sidebar.number_input("Rampa VDF [rpm/s] (eje motor)", min_value=1.0, value=300.0, step=10.0)
dt = st.sidebar.number_input("Paso de integración dt [s] (hidráulica)", min_value=0.001, value=0.01, step=0.01, format="%.3f")

row = df[tag_series == str(tag)].iloc[0]

# ---------------- 1) Entrada ----------------
st.header("1) Parámetros de entrada (datos)")
cA, cB, cC = st.columns(3)

with cA:
    st.subheader("Motor / Transmisión")
    st.write(f"**Potencia motor:** {row.get(col_power_kw, np.nan)} kW")
    if col_poles: st.write(f"**Polos:** {int(row[col_poles]) if not pd.isna(row.get(col_poles)) else '—'}")
    st.write(f"**Relación r = ω_motor/ω_bomba:** {row.get(col_r, np.nan)}")
    st.write(f"**n_motor min–max [rpm]:** {row.get(col_nmot_min, np.nan)} – {row.get(col_nmot_max, np.nan)}")
    if (col_npum_min in df.columns) and (col_npum_max in df.columns):
        st.write(f"**n_bomba min–max [rpm]:** {row.get(col_npum_min, np.nan)} – {row.get(col_npum_max, np.nan)}")

with cB:
    st.subheader("Inercias (kg·m²)")
    st.write(f"**J_m (motor):** {row.get(col_jm, np.nan)}")
    st.write(f"**J_driver (polea + manguito conductor):** {row.get(col_jdrv, np.nan)}")
    st.write(f"**J_driven (polea + manguito conducido):** {row.get(col_jdrn, np.nan)}")
    st.write(f"**J_imp (impulsor):** {row.get(col_jimp, np.nan)}")

with cC:
    st.subheader("Impulsor / Hidráulica")
    if col_dimp_mm: st.write(f"**D_imp (mm):** {row.get(col_dimp_mm, np.nan)}")
    st.write(f"**H0 (m):** {row.get(col_H0, np.nan)}")
    st.write(f"**K (m·s⁻²):** {row.get(col_K, np.nan)}")
    st.write("**η(Q)=a+bQ+cQ²** (Q en m³/s)")
    st.write(f"a={row.get(col_eta_a, np.nan)}, b={row.get(col_eta_b, np.nan)}, c={row.get(col_eta_c, np.nan)}")
    st.write(f"**ρ (kg/m³):** {row.get(col_rho, np.nan)}")
    st.write(f"**n_ref (rpm):** {row.get(col_nref, np.nan)}")
    st.write(f"**Q rango (m³/h):** {row.get(col_Qmin_h, np.nan)} – {row.get(col_Qmax_h, np.nan)}")

# ---------------- 2) Inercia equivalente ----------------
st.header("2) Inercia equivalente reflejada al eje del motor")
st.markdown(r"""
\[
J_{\text{eq}} = J_m + J_{\text{driver}} + \frac{J_{\text{driven}} + J_{\text{imp}}}{r^2}
\]
""")
J_m   = float(row.get(col_jm, 0.0) or 0.0)
J_drv = float(row.get(col_jdrv, 0.0) or 0.0)
J_drn = float(row.get(col_jdrn, 0.0) or 0.0)
J_imp = float(row.get(col_jimp, 0.0) or 0.0)
r_tr  = float(row.get(col_r, np.nan) or np.nan)
J_eq = np.nan if (pd.isna(r_tr) or r_tr == 0) else J_m + J_drv + (J_drn + J_imp)/(r_tr**2)
st.info(f"**J_eq (kg·m²):** {np.round(J_eq, 6)}")

# ---------------- 3) Tiempo sin hidráulica ----------------
st.header("3) Tiempo de reacción **sin** hidráulica")
st.markdown(r"""
\[
\dot n_{\text{torque}} = \frac{60}{2\pi}\frac{T_{\text{nom}}}{J_{\text{eq}}},\quad
t_{\text{par}} = \frac{\Delta n}{\dot n_{\text{torque}}},\quad
t_{\text{rampa}} = \frac{\Delta n}{\text{rampa}_{\text{VDF}}},\quad
t_{\text{final,sin}} = \max(t_{\text{par}}, t_{\text{rampa}})
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
    if col_tnom_nm and not pd.isna(row.get(col_tnom_nm)):
        T_nom = float(row[col_tnom_nm])
    else:
        P_kw = float(row.get(col_power_kw, 0.0) or 0.0)
        n_for_t = float(row.get(col_nmot_max, n_fin) or n_fin or 1.0)
        T_nom = 9550.0 * P_kw / max(n_for_t, 1.0)
    st.number_input("Par disponible T_nom [Nm]", value=float(T_nom), step=10.0, key="tnom_view")

delta_n = max(0.0, n_fin - n_ini)
accel_torque = (60.0/(2.0*math.pi)) * (T_nom / max(J_eq, 1e-9))
t_par   = delta_n / max(accel_torque, 1e-9)
t_rampa = delta_n / max(ramp_rpm_s, 1e-9)
t_final_nohyd = max(t_par, t_rampa)
st.success(f"Δn={delta_n:.1f} rpm | ẋ_n={accel_torque:.1f} rpm/s  →  t_par={t_par:.2f} s, t_rampa={t_rampa:.2f} s, **t_final={t_final_nohyd:.2f} s**")

# ---------------- 4) Curva de sistema + hidráulica ----------------
st.header("4) Curva de sistema y tiempo **con** hidráulica")
H0   = float(row.get(col_H0, np.nan) or np.nan)
K    = float(row.get(col_K,  np.nan) or np.nan)
eta_a = float(row.get(col_eta_a, 0.0) or 0.0)
eta_b = float(row.get(col_eta_b, 0.0) or 0.0)
eta_c = float(row.get(col_eta_c, 0.0) or 0.0)
rho   = float(row.get(col_rho, 1000.0) or 1000.0)

def eta_of_Q(Q_m3s): return eta_a + eta_b*Q_m3s + eta_c*(Q_m3s**2)
def H_of_Q(Q_m3s):   return H0 + K*(Q_m3s**2)

n_ref = float(row.get(col_nref, n_fin) or n_fin or 1.0)
Qmin_h = float(row.get(col_Qmin_h, 0.0) or 0.0)
Qmax_h = float(row.get(col_Qmax_h, 0.0) or 0.0)
Qref_h = (Qmin_h + Qmax_h)/2.0 if (Qmax_h > 0) else 0.0
alpha_default = (Qref_h / max(n_ref, 1.0))
alpha_user = st.number_input("α: caudal por rpm de bomba [m³/h·rpm]", value=float(alpha_default), step=1.0)
alpha = alpha_user / 3600.0   # m³/s por rpm

Qh = np.linspace(max(Qmin_h*0.5, 1.0), max(Qmax_h*1.2, max(50.0, Qref_h*1.5)), 100)
Qs = Qh/3600.0
fig = go.Figure()
fig.add_trace(go.Scatter(x=Qh, y=H_of_Q(Qs), mode="lines", name="H(Q)=H0+KQ²", line=dict(width=3)))
fig.update_layout(xaxis_title="Q [m³/h]", yaxis_title="H [m]", template="plotly_white", height=420)
st.plotly_chart(fig, use_container_width=True)

st.markdown(r"""
\[
T_{\text{pump}}(Q,n)=\frac{\rho g Q \left(H_0 + KQ^2\right)}{\eta(Q)\,\omega(n)},\quad
\omega(n)=2\pi n/60,\quad T_{\text{motor}}=\frac{T_{\text{pump}}}{r}
\]
\[
\frac{dn_{\text{motor}}}{dt}=\min\!\left(\text{rampa}_{\text{VDF}},\;\frac{60}{2\pi}\frac{T_{\text{nom}}-T_{\text{motor}}(n)}{J_{\text{eq}}}\right)
\]
""")

def simulate_with_hyd(n0, n1, dt, T_nom, J_eq, r_tr, alpha_m3s_per_rpm, H0, K, rho, ramp_rpm_s,
                      eta_a, eta_b, eta_c, nmax_iter=2_000_000):
    t = 0.0
    n = n0
    g = 9.80665
    times, n_list, q_list, tload_list = [0.0], [n0], [0.0], [0.0]
    while n < n1 and len(times) < nmax_iter:
        n_cmd = min(n1, n + ramp_rpm_s*dt)
        n_bomba = n / max(r_tr, 1e-9)
        Q = max(0.0, alpha_m3s_per_rpm * n_bomba)
        eta_q = clamp(eta_of_Q(Q), 0.3, 0.9)
        H_q = H_of_Q(Q)
        omega_b = 2.0*math.pi*max(n_bomba, 1e-6)/60.0

        P_h = rho * g * Q * H_q
        T_pump = (P_h / max(eta_q, 1e-6)) / omega_b
        T_motor_load = T_pump / max(r_tr, 1e-9)

        accel_torque = (60.0/(2.0*math.pi)) * ((T_nom - T_motor_load) / max(J_eq, 1e-9))  # rpm/s
        accel = min(ramp_rpm_s, max(accel_torque, 0.0))
        n = min(n_cmd, n + accel*dt)
        t += dt

        times.append(t); n_list.append(n); q_list.append(Q*3600.0); tload_list.append(T_motor_load)
        if n >= n1 - 1e-6: break

    return pd.DataFrame({"t_s": times, "n_motor_rpm": n_list, "Q_m3h": q_list, "T_load_motor_Nm": tload_list})

if not any(pd.isna([J_eq, r_tr, H0, K, rho])):
    sim = simulate_with_hyd(n_ini, n_fin, dt, T_nom, J_eq, r_tr, alpha, H0, K, rho, ramp_rpm_s,
                            eta_a, eta_b, eta_c)
    t_hyd = float(sim["t_s"].iloc[-1]) if len(sim) else np.nan

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

# ---------------- 5) Exportar resumen ----------------
st.header("5) Exportar resultados")
summary = {
    "TAG": str(tag),
    "J_m_kgm2": J_m, "J_driver_kgm2": J_drv, "J_driven_kgm2": J_drn, "J_imp_kgm2": J_imp,
    "r_trans": r_tr, "J_eq_kgm2": J_eq,
    "T_nom_Nm": T_nom, "rampa_rpm_s": ramp_rpm_s,
    "n_ini_rpm": float(n_ini), "n_fin_rpm": float(n_fin), "delta_n_rpm": float(n_fin-n_ini),
    "accel_rpm_s_torque": accel_torque, "t_par_s": t_par, "t_rampa_s": t_rampa,
    "t_final_sin_hidraulica_s": t_final_nohyd,
    "H0_m": H0, "K_m_s2": K, "rho_kgm3": rho,
    "eta_a": eta_a, "eta_b": eta_b, "eta_c": eta_c,
}
summary_df = pd.DataFrame([summary])
st.dataframe(summary_df, use_container_width=True)
st.download_button(
    "Descargar resumen (CSV)",
    data=summary_df.to_csv(index=False).encode("utf-8"),
    file_name=f"resumen_{tag}.csv",
    mime="text/csv"
)
