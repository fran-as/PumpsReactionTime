# app.py
# ─────────────────────────────────────────────────────────────────────────────
# Dashboard: Tiempo de reacción de bombas con VDF
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# Configuración general
# =============================================================================

st.set_page_config(
    page_title="Memoria de Cálculo – Tiempo de reacción (VDF)",
    layout="wide",
)

# Colores
BLUE = "#1f77b4"   # Dado (dataset) → azul
GREEN = "#2ca02c"  # Calculado      → verde
GRAY = "#6c757d"

# Utilidades ------------------------------------------------------------------

def dataset_path() -> Path:
    return Path(__file__).with_name("dataset.csv")

def images_path(name: str) -> Path:
    return Path(__file__).with_name("images") / name

def get_num(x) -> float:
    if x is None:
        return float("nan")
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace(" ", "").replace("\u00a0", "")
    s = s.replace(",", ".")
    import re
    s = re.sub(r"[^0-9eE\+\-\.]", "", s)
    try:
        return float(s)
    except Exception:
        return float("nan")

def fmt_num(x, unit: str = "", ndigits: int = 2) -> str:
    """Formatea números con coma decimal; si es texto, lo devuelve tal cual."""
    if isinstance(x, str):
        return f"{x} {unit}".strip()
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    s = f"{x:,.{ndigits}f}"
    s = s.replace(",", "_").replace(".", ",").replace("_", ".")
    return f"{s} {unit}".strip()

def color_value(text: str, color: str = BLUE, bold: bool = True) -> str:
    w = "600" if bold else "400"
    return f'<span style="color:{color}; font-weight:{w}">{text}</span>'

def val_blue(x, unit="", ndigits=2) -> str:
    return color_value(fmt_num(x, unit, ndigits), BLUE)

def val_green(x, unit="", ndigits=2) -> str:
    return color_value(fmt_num(x, unit, ndigits), GREEN)

# === MAPEOS DESDE dataset.csv (sep=';' decimal=',') ===
ATTR = {
    # Identificación / rating
    "TAG":                 {"col": "TAG",                   "unit": "",      "type": "str"},
    "pump_model":          {"col": "pumpmodel",             "unit": "",      "type": "str"},
    "impeller_d_mm":       {"col": "impeller_d_mm",         "unit": "mm",    "type": "num"},
    "motorpower_kw":       {"col": "motorpower_kw",         "unit": "kW",    "type": "num"},
    "t_nom_nm":            {"col": "t_nom_nm",              "unit": "N·m",   "type": "num"},
    "r":                   {"col": "r_trans",               "unit": "",      "type": "num"},
    "n_m_min":             {"col": "motor_n_min_rpm",       "unit": "rpm",   "type": "num"},
    "n_m_max":             {"col": "motor_n_max_rpm",       "unit": "rpm",   "type": "num"},
    "n_p_min":             {"col": "pump_n_min_rpm",        "unit": "rpm",   "type": "num"},
    "n_p_max":             {"col": "pump_n_max_rpm",        "unit": "rpm",   "type": "num"},

    # Inercias (catálogo TB Woods / Metso)  -------------------------
    "J_m":                 {"col": "motor_j_kgm2",          "unit": "kg·m²", "type": "num"},
    "J_driver":            {"col": "driverpulley_j_kgm2",   "unit": "kg·m²", "type": "num"},
    "J_sleeve_driver":     {"col": "driverbushing_j_kgm2",  "unit": "kg·m²", "type": "num"},  # ≈10% de J_driver si falta
    "J_driven":            {"col": "drivenpulley_j_Kgm2",   "unit": "kg·m²", "type": "num"},
    "J_sleeve_driven":     {"col": "drivenbushing_j_Kgm2",  "unit": "kg·m²", "type": "num"},  # ≈10% de J_driven si falta
    "J_imp":               {"col": "impeller_j_kgm2",       "unit": "kg·m²", "type": "num"},

    # Hidráulica / pulpa
    "H0_m":                {"col": "H0_m",                  "unit": "m",     "type": "num"},
    "K_H_per_Q":           {"col": "K_H_per_Q",             "unit": "m/(m³/h)", "type": "num"},
    "Qmin_m3h":            {"col": "Qmin_m3h",              "unit": "m³/h",  "type": "num"},
    "Qbest_m3h":           {"col": "Qbest_m3h",             "unit": "m³/h",  "type": "num"},
    "Qmax_m3h":            {"col": "Qmax_m3h",              "unit": "m³/h",  "type": "num"},
    "eta":                 {"col": "eta",                   "unit": "",      "type": "num"},
    "SlurryDensity":       {"col": "SlurryDensity_Kgm3",    "unit": "kg/m³", "type": "num"},
}

def get_val(row, key, default=np.nan):
    meta = ATTR[key]
    col = meta["col"]
    v = row.get(col, default)
    if meta["type"] == "num":
        try:
            return float(v)
        except Exception:
            return default
    return v

def val_blue(x, unit=""):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "<span style='color:#1f77b4'>—</span>"
    u = f" {unit}" if unit else ""
    return f"<span style='color:#1f77b4'><b>{x:,.2f}</b>{u}</span>"

def val_green(x, unit=""):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "<span style='color:#2ca02c'>—</span>"
    u = f" {unit}" if unit else ""
    return f"<span style='color:#2ca02c'><b>{x:,.2f}</b>{u}</span>"

# =============================================================================
# Encabezado con logos + título
# =============================================================================

colL, colC, colR = st.columns([1, 3, 1])
with colL:
    p = images_path("metso_logo.png")
    if p.exists():
        st.image(str(p), use_container_width=True)
with colC:
    st.markdown(
        "<h1 style='text-align:center; margin-top: 0.2rem'>Tiempo de reacción de bombas con VDF</h1>",
        unsafe_allow_html=True,
    )
with colR:
    p = images_path("ausenco_logo.png")
    if p.exists():
        st.image(str(p), use_container_width=True)

st.markdown("---")

# =============================================================================
# Dataset + selector de TAG
# =============================================================================

df_raw = load_data()
tag_col = col_lookup(df_raw, "tag") or "TAG"
tags = df_raw[tag_col].astype(str).tolist()
tag_sel = st.sidebar.selectbox("Selecciona TAG", tags, index=0)
row = df_raw[df_raw[tag_col].astype(str) == str(tag_sel)].iloc[0]

def get_val(key: str, default=np.nan):
    col = col_lookup(df_raw, key)
    if col and col in row:
        return get_num(row[col])
    return default

txt_pump_model = row.get(col_lookup(df_raw, "pump_model") or "pumpmodel", "—")

P_motor_kW = get_val("motor_power_kw")
T_nom_nm   = get_val("t_nom_nm")
r_nm_np    = get_val("r_nm_np")
J_m        = get_val("J_m")
J_driver   = get_val("J_driver")
J_driven   = get_val("J_driven")
J_imp      = get_val("J_imp")
D_imp_mm   = get_val("impeller_d_mm")

n_ref_rpm  = get_val("n_ref_rpm")
if np.isnan(n_ref_rpm):
    n_ref_rpm = 1000.0
n_p_min = 0.5 * n_ref_rpm
n_p_max = 1.0 * n_ref_rpm
if not np.isnan(r_nm_np) and r_nm_np > 0:
    n_m_min = n_p_min * r_nm_np
    n_m_max = n_p_max * r_nm_np
else:
    n_m_min = 2 * n_p_min
    n_m_max = 2 * n_p_max

H0_m     = get_val("H0_m")
K_m_s2   = get_val("K_m_s2")
eta_a    = get_val("eta_a")
eta_b    = get_val("eta_b")
eta_c    = get_val("eta_c")
eta_min  = get_val("eta_min_clip", 0.4)
eta_max  = get_val("eta_max_clip", 0.88)
eta_beta = get_val("eta_beta")
Q_ref    = get_val("Q_ref_m3h")
Q_min_ds = get_val("Q_min_m3h")
Q_max_ds = get_val("Q_max_m3h")
rho      = get_val("rho_kgm3")
if np.isnan(rho) or rho <= 0:
    rho = 1000.0
g = 9.81

# =============================================================================
# 1) Parámetros
# =============================================================================

st.markdown("## 1) Parámetros")

c1, c2, c3 = st.columns([1.2, 1.2, 1.2])

with c1:
    st.markdown("**Identificación**")
    st.markdown(f"- Modelo de bomba: {val_blue(txt_pump_model, '', 0)}", unsafe_allow_html=True)
    st.markdown(f"- TAG: {val_blue(tag_sel, '', 0)}", unsafe_allow_html=True)
    st.markdown(f"- Diámetro impulsor: {val_blue(D_imp_mm, 'mm')}", unsafe_allow_html=True)

with c2:
    st.markdown("**Motor & transmisión**")
    st.markdown(f"- Potencia motor instalada: {val_blue(P_motor_kW, 'kW')}", unsafe_allow_html=True)
    st.markdown(f"- Par nominal del motor: {val_blue(T_nom_nm, 'N·m')}", unsafe_allow_html=True)
    st.markdown(f"- Relación transmisión (r = n_motor/n_bomba): {val_blue(r_nm_np, '')}", unsafe_allow_html=True)
    st.markdown(f"- Velocidad motor min–max: {val_blue(n_m_min, 'rpm', 0)} – {val_blue(n_m_max, 'rpm', 0)}", unsafe_allow_html=True)

with c3:
    st.markdown("**Bomba (25–50 Hz por afinidad)**")
    st.markdown(f"- Velocidad bomba min–max: {val_blue(n_p_min, 'rpm', 0)} – {val_blue(n_p_max, 'rpm', 0)}", unsafe_allow_html=True)
    st.markdown(f"- Caudal de referencia Q_ref: {val_blue(Q_ref, 'm³/h', 0)}", unsafe_allow_html=True)
    st.markdown(f"- Densidad de pulpa ρ: {val_blue(rho, 'kg/m³', 0)}", unsafe_allow_html=True)

st.markdown("---")

# ────────────────────────────────────────────────────────────────────────────
# 2) Cálculo de inercia equivalente
# ────────────────────────────────────────────────────────────────────────────
st.header("2) Cálculo de inercia equivalente")

colL, colR = st.columns([1.1, 1])

with colL:
    st.subheader("Inercias individuales")

    J_m = get_val(row, "J_m", 0.0)
    J_driver = get_val(row, "J_driver", 0.0)
    J_sleeve_driver = get_val(row, "J_sleeve_driver", np.nan)
    if np.isnan(J_sleeve_driver):  # fallback 10% catálogo TB Woods
        J_sleeve_driver = 0.10 * J_driver

    J_driven = get_val(row, "J_driven", 0.0)
    J_sleeve_driven = get_val(row, "J_sleeve_driven", np.nan)
    if np.isnan(J_sleeve_driven):  # fallback 10%
        J_sleeve_driven = 0.10 * J_driven

    J_imp = get_val(row, "J_imp", 0.0)
    r     = max(get_val(row, "r", 1.0), 1e-6)

    st.markdown(f"- Motor (J_m): {val_blue(J_m, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Polea motriz (J_driver): {val_blue(J_driver, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Manguito motriz (J_sleeve_driver≈10% J_driver): {val_blue(J_sleeve_driver, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Polea conducida (J_driven): {val_blue(J_driven, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Manguito conducido (J_sleeve_driven≈10% J_driven): {val_blue(J_sleeve_driven, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Impulsor/rotor de bomba (J_imp): {val_blue(J_imp, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Relación r (n_m/n_p): {val_blue(r, '')}", unsafe_allow_html=True)

    # J_eq en el eje del motor (inercias del lado bomba vistas /r²)
    J_eq = (J_m + J_driver + J_sleeve_driver) + (J_driven + J_sleeve_driven + J_imp) / (r**2)

    st.markdown(f"**Inercia equivalente (J_eq):** {val_green(J_eq, 'kg·m²')}", unsafe_allow_html=True)

with colR:
    st.subheader("Fórmula utilizada")
    st.latex(r"J_{\mathrm{eq}} \;=\; J_m \;+\; J_{\mathrm{driver}} \;+\; J_{\mathrm{sleeve,driver}} \;+\; \frac{J_{\mathrm{driven}} + J_{\mathrm{sleeve,driven}} + J_{\mathrm{imp}}}{r^2}")
    with st.expander("Formulación de las inercias por componente", expanded=True):
        st.markdown(
            "- **Motor** (J_m): obtenidas de hojas de datos **WEG**.\n"
            "- **Poleas** (J_driver, J_driven): obtenidas de catálogo **TB Woods**.\n"
            "- **Manguitos** (J_sleeve_driver, J_sleeve_driven): si no hay dato, se aproxima **10%** de la inercia de su polea.\n"
            "- **Impulsor** (J_imp): de manuales **Metso**.\n\n"
            "Las inercias del lado bomba giran a \(\\omega_p=\\omega_m/r\\). "
            "Igualando energías cinéticas a una \(\\omega_m\\) común se obtiene la división por \(r^2\) para términos del lado de la bomba."
        )

# =============================================================================
# 3) Tiempo inercial (par vs rampa)
# =============================================================================

st.markdown("## 3) Tiempo inercial (par disponible vs rampa VDF)")

rampa_vdf = st.slider("Rampa VDF en el motor [rpm/s]", min_value=10, max_value=400, value=100, step=5)

n_dot_torque = float("nan")
if not (np.isnan(J_eq) or np.isnan(T_nom_nm) or J_eq <= 0):
    n_dot_torque = (60.0 / (2.0 * math.pi)) * (T_nom_nm / J_eq)  # rpm/s
    t_par = (n_m_max - n_m_min) / max(n_dot_torque, 1e-9)        # s
else:
    t_par = float("nan")

t_rampa = (n_m_max - n_m_min) / max(rampa_vdf, 1e-9)

cA, cB, cC = st.columns(3)
with cA:
    st.markdown(f"- Aceleración por par (lado motor): {val_green(n_dot_torque, 'rpm/s')}", unsafe_allow_html=True)
with cB:
    st.markdown(f"- Tiempo por par (25→50 Hz): {val_green(t_par, 's')}", unsafe_allow_html=True)
with cC:
    st.markdown(f"- Tiempo por rampa VDF: {val_blue(t_rampa, 's')}", unsafe_allow_html=True)

def pill(text: str, bg: str = "#e8f5e9", color: str = "#1b5e20"):
    st.markdown(
        f"""
        <div style="border-left: 5px solid {color}; background:{bg}; padding:0.8rem 1rem; border-radius:0.5rem; margin-top:0.5rem">
            <b style="color:{color}">{text}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

lim3 = "Tiempo limitante (sección 3): "
if not np.isnan(t_par) and t_par > t_rampa:
    pill(lim3 + f"**por par** = {fmt_num(t_par, 's')}")
else:
    pill(lim3 + f"**por rampa VDF** = {fmt_num(t_rampa, 's')}")

st.markdown("---")

# =============================================================================
# 4) Integración con carga hidráulica
# =============================================================================

st.markdown("## 4) Integración con carga hidráulica")
st.latex(r"H(Q) \;=\; H_0 \;+\; K\,\left(\dfrac{Q}{3600}\right)^2 \qquad \big[\,Q:\ \mathrm{m^3/h},\ H:\ \mathrm{m}\,\big]")

c4a, c4b, c4c, c4d = st.columns(4)
with c4a:
    st.markdown(f"- H₀: {val_blue(H0_m, 'm')}", unsafe_allow_html=True)
with c4b:
    st.markdown(f"- K: {val_blue(K_m_s2, 'm·s²')}", unsafe_allow_html=True)
with c4c:
    st.markdown(f"- ρ: {val_blue(rho, 'kg/m³', 0)}", unsafe_allow_html=True)
with c4d:
    st.markdown(f"- η (clip): {val_blue(eta_min*100, '%', 0)} – {val_blue(eta_max*100, '%', 0)}", unsafe_allow_html=True)

# Rango de caudales
if np.isnan(Q_min_ds) or np.isnan(Q_max_ds) or Q_min_ds >= Q_max_ds:
    if not np.isnan(Q_ref) and Q_ref > 0:
        qmin_default = 0.6 * Q_ref
        qmax_default = 1.1 * Q_ref
    else:
        qmin_default = 100.0
        qmax_default = 300.0
else:
    qmin_default, qmax_default = Q_min_ds, Q_max_ds

Q_min, Q_max = st.slider(
    "Rango de caudal considerado [m³/h]",
    min_value=0.0, max_value=max(5000.0, qmax_default * 1.2),
    value=(float(qmin_default), float(qmax_default)),
    step=1.0,
)

def Q_from_np(n_p: np.ndarray) -> np.ndarray:
    if n_p_max <= n_p_min + 1e-9:
        return np.full_like(n_p, Q_min)
    return Q_min + (Q_max - Q_min) * (n_p - n_p_min) / (n_p_max - n_p_min)

def eta_from_Q(Q_m3h: np.ndarray) -> np.ndarray:
    if not np.isnan(eta_beta) and eta_beta > 0 and not np.isnan(eta_min) and not np.isnan(eta_max):
        e = np.full_like(Q_m3h, float(eta_beta))
    elif not (np.isnan(eta_a) or np.isnan(eta_b) or np.isnan(eta_c) or np.isnan(Q_ref) or Q_ref <= 0):
        beta = Q_m3h / Q_ref
        e = eta_a + eta_b * beta + eta_c * (beta**2)
    else:
        e = np.full_like(Q_m3h, 0.72)
    emin = eta_min if not np.isnan(eta_min) else 0.4
    emax = eta_max if not np.isnan(eta_max) else 0.88
    return np.clip(e, emin, emax)

# Mallas
N = 600
n_m_grid = np.linspace(n_m_min, n_m_max, N)
n_p_grid = n_m_grid / max(r_nm_np, 1e-9)
Q_grid   = Q_from_np(n_p_grid)
q_grid   = Q_grid / 3600.0
H_grid   = H0_m + K_m_s2 * (q_grid**2)
eta_grid = eta_from_Q(Q_grid)

# Potencias y pares
P_h_grid = rho * g * q_grid * H_grid / np.maximum(eta_grid, 1e-6)   # W
omega_p  = 2.0 * math.pi * n_p_grid / 60.0                          # rad/s
T_pump   = np.where(omega_p > 1e-9, P_h_grid / omega_p, 0.0)        # N·m (eje bomba)
T_load_m = T_pump / max(r_nm_np, 1e-9)                              # N·m (eje motor)
T_disp_m = np.full_like(T_load_m, T_nom_nm)

# Gráficos
fig_t = go.Figure()
fig_t.add_trace(go.Scatter(x=n_p_grid, y=T_load_m, name="Par resistente reflejado (motor)", mode="lines"))
fig_t.add_trace(go.Scatter(x=n_p_grid, y=T_disp_m, name="Par motor disponible", mode="lines"))
fig_t.update_layout(
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis_title="Velocidad bomba n_p [rpm]",
    yaxis_title="Par en el eje motor [N·m]",
    legend=dict(orientation="h", y=1.02, yanchor="bottom"),
    height=360,
)
st.plotly_chart(fig_t, use_container_width=True)

fig_p = go.Figure()
fig_p.add_trace(go.Scatter(x=n_p_grid, y=P_h_grid/1000.0, mode="lines", name="P hidráulica [kW]"))
fig_p.update_layout(
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis_title="Velocidad bomba n_p [rpm]",
    yaxis_title="Potencia hidráulica [kW]",
    legend=dict(orientation="h"),
    height=320,
)
st.plotly_chart(fig_p, use_container_width=True)

# Integración temporal con carga
if not (np.isnan(J_eq) or J_eq <= 0):
    omega_m_grid = 2.0 * math.pi * n_m_grid / 60.0
    d_omega = np.diff(omega_m_grid)
    T_net = T_disp_m - T_load_m
    T_net_clip = np.maximum(T_net, 1e-6)
    dt = (J_eq * d_omega) / T_net_clip[:-1]
    t_hyd = float(np.sum(dt))
else:
    t_hyd = float("nan")

c4_1, c4_2 = st.columns(2)
with c4_1:
    st.markdown(f"- Tiempo por carga **hidráulica** (integración): {val_green(t_hyd, 's')}", unsafe_allow_html=True)
with c4_2:
    t_lim_4 = max(t_hyd, t_rampa) if not np.isnan(t_hyd) else t_rampa
    which = "hidráulica" if (not np.isnan(t_hyd) and t_hyd > t_rampa) else "rampa VDF"
    pill(f"Tiempo limitante (sección 4): **{which} = {fmt_num(t_lim_4, 's')}**")

with st.expander("Detalles de la formulación (sección 4)"):
    st.markdown(
        "- **Curva del sistema:** $H(Q)=H_0 + K\\,(Q/3600)^2$.\n"
        "- **Afinidad:** $Q\\propto n_p$ en 25–50 Hz.\n"
        "- **Potencia hidráulica:** $P_h = \\rho g\\,Q_s\\,H(Q)/\\eta(Q)$, con $Q_s=Q/3600$.\n"
        "- **Par de bomba:** $T_{pump} = P_h/\\omega_p$, $\\omega_p=2\\pi n_p/60$.\n"
        "- **Reflejo al motor:** $T_{load,m}=T_{pump}/r$.\n"
        "- **Dinámica motor:** $\\dot\\omega_m=(T_{disp}-T_{load,m})/J_{eq}$; $\\Delta t = J_{eq}\\,\\Delta\\omega_m/(T_{disp}-T_{load,m})$.\n"
    )
