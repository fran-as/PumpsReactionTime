# app.py
# ─────────────────────────────────────────────────────────────────────────────
# Dashboard: Tiempo de reacción de bombas con VDF (25–50 Hz)
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# Configuración general
# =============================================================================

st.set_page_config(page_title="Memoria de Cálculo – Tiempo de reacción (VDF)", layout="wide")

# Colores para valores
BLUE = "#1f77b4"   # parámetros dados (dataset)
GREEN = "#2ca02c"  # resultados calculados

# -----------------------------------------------------------------------------
# Utilidades de paths
# -----------------------------------------------------------------------------
def dataset_path() -> Path:
    return Path(__file__).with_name("dataset.csv")

def images_path(name: str) -> Path:
    return Path(__file__).with_name("images") / name

# -----------------------------------------------------------------------------
# Parsing numérico robusto (decimal ',' aceptado)
# -----------------------------------------------------------------------------
def to_float(x) -> float:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return float("nan")
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace(" ", "").replace("\u00a0", "").replace(",", ".")
    import re
    s = re.sub(r"[^0-9eE\+\-\.]", "", s)
    try:
        return float(s)
    except Exception:
        return float("nan")

def fmt_num(x, unit: str = "", ndigits: int = 2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    s = f"{x:,.{ndigits}f}".replace(",", "_").replace(".", ",").replace("_", ".")
    return f"{s} {unit}".strip()

def span_color(text: str, color: str, bold=True) -> str:
    w = "600" if bold else "400"
    return f"<span style='color:{color}; font-weight:{w}'>{text}</span>"

def val_blue(x, unit="", ndigits=2) -> str:
    return span_color(fmt_num(x, unit, ndigits), BLUE)

def val_green(x, unit="", ndigits=2) -> str:
    return span_color(fmt_num(x, unit, ndigits), GREEN)

# -----------------------------------------------------------------------------
# Mapeo de columnas del dataset (CSV: sep=';' decimal=',')
# -----------------------------------------------------------------------------
ATTR = {
    "TAG":                 {"col": "TAG",                   "type": "str"},
    "pump_model":          {"col": "pumpmodel",             "type": "str"},
    "impeller_d_mm":       {"col": "impeller_d_mm",         "type": "num"},
    "motorpower_kw":       {"col": "motorpower_kw",         "type": "num"},
    "t_nom_nm":            {"col": "t_nom_nm",              "type": "num"},
    "r":                   {"col": "r_trans",               "type": "num"},

    "n_m_min":             {"col": "motor_n_min_rpm",       "type": "num"},
    "n_m_max":             {"col": "motor_n_max_rpm",       "type": "num"},
    "n_p_min":             {"col": "pump_n_min_rpm",        "type": "num"},
    "n_p_max":             {"col": "pump_n_max_rpm",        "type": "num"},

    # Inercias (TB Woods / Metso)
    "J_m":                 {"col": "motor_j_kgm2",          "type": "num"},
    "J_driver":            {"col": "driverpulley_j_kgm2",   "type": "num"},
    "J_bushing_driver":    {"col": "driverbushing_j_kgm2",  "type": "num"},  # ≈10% fallback
    "J_driven":            {"col": "drivenpulley_j_kgm2",   "type": "num"},
    "J_bushing_driven":    {"col": "drivenbushing_j_kgm2",  "type": "num"},  # ≈10% fallback
    "J_imp":               {"col": "impeller_j_kgm2",       "type": "num"},

    # Hidráulica / eficiencia
    "H0_m":                {"col": "H0_m",                  "type": "num"},
    "K_m_s2":              {"col": "K_m_s2",                "type": "num"},
    "Q_min_m3h":           {"col": "Q_min_m3h",             "type": "num"},
    "Q_max_m3h":           {"col": "Q_max_m3h",             "type": "num"},
    "Q_ref_m3h":           {"col": "Q_ref_m3h",             "type": "num"},
    "n_ref_rpm":           {"col": "n_ref_rpm",             "type": "num"},
    "eta":                 {"col": "eta_beta",              "type": "num"},   # valor fijo opcional
    "eta_a":               {"col": "eta_a",                 "type": "num"},
    "eta_b":               {"col": "eta_b",                 "type": "num"},
    "eta_c":               {"col": "eta_c",                 "type": "num"},
    "eta_min":             {"col": "eta_min_clip",          "type": "num"},
    "eta_max":             {"col": "eta_max_clip",          "type": "num"},
    "rho":                 {"col": "SlurryDensity_kgm3",    "type": "num"},
}

# -----------------------------------------------------------------------------
# Carga del dataset
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(dataset_path(), sep=";", decimal=",", dtype=str)
    for meta in ATTR.values():
        c = meta["col"]
        if c in df.columns and meta["type"] == "num":
            df[c] = df[c].map(to_float)
    return df

# -----------------------------------------------------------------------------
# Encabezado con logos y título
# -----------------------------------------------------------------------------
colL, colC, colR = st.columns([1, 3, 1])
with colL:
    p = images_path("metso_logo.png")
    if p.exists():
        st.image(str(p), use_container_width=True)
with colC:
    st.markdown(
        "<h1 style='text-align:center; margin-top:0.2rem'>Tiempo de reacción de bombas con VDF (25–50 Hz)</h1>",
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
df = load_data()
tags = df[ATTR["TAG"]["col"]].astype(str).tolist()
tag_sel = st.sidebar.selectbox("Selecciona TAG", tags, index=0)
row = df[df[ATTR["TAG"]["col"]].astype(str) == str(tag_sel)].iloc[0]

def rv(key: str, default=np.nan):
    col = ATTR[key]["col"]
    return row[col] if ATTR[key]["type"] == "str" else to_float(row[col]) if col in row else default

# Valores base
pump_model = rv("pump_model", "—")
D_imp_mm   = rv("impeller_d_mm")
P_motor_kW = rv("motorpower_kw")
T_nom_nm   = rv("t_nom_nm")
r_nm_np    = rv("r")

n_m_min = rv("n_m_min"); n_m_max = rv("n_m_max")
n_p_min = rv("n_p_min"); n_p_max = rv("n_p_max")

# Si faltan, reconstruir con r
if (math.isnan(n_p_min) or math.isnan(n_p_max)) and not math.isnan(n_m_min) and not math.isnan(r_nm_np) and r_nm_np > 0:
    n_p_min = n_m_min / r_nm_np
    n_p_max = n_m_max / r_nm_np
if (math.isnan(n_m_min) or math.isnan(n_m_max)) and not math.isnan(n_p_min) and not math.isnan(r_nm_np):
    n_m_min = n_p_min * r_nm_np
    n_m_max = n_p_max * r_nm_np

# Hidráulica
H0_m   = rv("H0_m")
K_m_s2 = rv("K_m_s2")
Q_min_ds = rv("Q_min_m3h")
Q_max_ds = rv("Q_max_m3h")
Q_ref  = rv("Q_ref_m3h")
n_ref  = rv("n_ref_rpm")
rho    = rv("rho"); rho = 1000.0 if math.isnan(rho) or rho <= 0 else rho
eta_fixed = rv("eta")
eta_a = rv("eta_a"); eta_b = rv("eta_b"); eta_c = rv("eta_c")
eta_min = rv("eta_min"); eta_max = rv("eta_max")
g = 9.81

# =============================================================================
# 1) Parámetros
# =============================================================================
st.markdown("## 1) Parámetros")

c1, c2, c3 = st.columns([1.2, 1.2, 1.2])
with c1:
    st.markdown("**Identificación**")
    st.markdown(f"- Modelo de bomba: {val_blue(pump_model, '', 0)}", unsafe_allow_html=True)
    st.markdown(f"- TAG: {val_blue(tag_sel, '', 0)}", unsafe_allow_html=True)
    st.markdown(f"- Diámetro de impulsor: {val_blue(D_imp_mm, 'mm')}", unsafe_allow_html=True)

with c2:
    st.markdown("**Motor & transmisión**")
    st.markdown(f"- Potencia motor instalada: {val_blue(P_motor_kW, 'kW')}", unsafe_allow_html=True)
    st.markdown(f"- Par nominal del motor: {val_blue(T_nom_nm, 'N·m')}", unsafe_allow_html=True)
    st.markdown(f"- Relación $r=n_m/n_p$: {val_blue(r_nm_np, '')}", unsafe_allow_html=True)
    st.markdown(f"- Velocidad motor min–max: {val_blue(n_m_min, 'rpm', 0)} – {val_blue(n_m_max, 'rpm', 0)}", unsafe_allow_html=True)

with c3:
    st.markdown("**Bomba (25–50 Hz por afinidad)**")
    st.markdown(f"- Velocidad bomba min–max: {val_blue(n_p_min, 'rpm', 0)} – {val_blue(n_p_max, 'rpm', 0)}", unsafe_allow_html=True)
    st.markdown(f"- Densidad de pulpa $\\rho$: {val_blue(rho, 'kg/m³', 0)}", unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# 2) Cálculo de inercia equivalente
# =============================================================================
st.header("2) Cálculo de inercia equivalente")
colL, colR = st.columns([1.1, 1])

with colL:
    st.subheader("Inercias individuales")
    J_m       = rv("J_m")
    J_driver  = rv("J_driver")
    J_bdrv    = rv("J_bushing_driver")
    if math.isnan(J_bdrv) and not math.isnan(J_driver):  # fallback 10%
        J_bdrv = 0.10 * J_driver
    J_driven  = rv("J_driven")
    J_bdrn    = rv("J_bushing_driven")
    if math.isnan(J_bdrn) and not math.isnan(J_driven):  # fallback 10%
        J_bdrn = 0.10 * J_driven
    J_imp     = rv("J_imp")
    r         = r_nm_np if not math.isnan(r_nm_np) and r_nm_np > 0 else 1.0

    st.markdown(f"- Motor (J_m): {val_blue(J_m, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Polea motriz (J_driver): {val_blue(J_driver, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Bushing motriz (≈10% de J_driver): {val_blue(J_bdrv, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Polea conducida (J_driven): {val_blue(J_driven, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Bushing conducido (≈10% de J_driven): {val_blue(J_bdrn, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Impulsor de bomba (J_imp): {val_blue(J_imp, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Relación $r=n_m/n_p$: {val_blue(r, '')}", unsafe_allow_html=True)

    J_eq = (J_m + J_driver + J_bdrv) + (J_driven + J_bdrn + J_imp) / (r**2)
    st.markdown(f"**Inercia equivalente (J_eq):** {val_green(J_eq, 'kg·m²')}", unsafe_allow_html=True)

with colR:
    st.subheader("Fórmula utilizada")
    st.latex(r"J_{\mathrm{eq}} = J_m + J_{\mathrm{driver}} + J_{\mathrm{bushing,driver}} + \frac{J_{\mathrm{driven}}+J_{\mathrm{bushing,driven}}+J_{\mathrm{imp}}}{r^2}")
    with st.expander("Formulación de las inercias por componente", expanded=True):
        st.markdown(
            "- **Motor** \(J_m\): hojas de datos **WEG**.\n"
            "- **Poleas** \(J_{driver}, J_{driven}\): catálogo **TB Woods**.\n"
            "- **Bushing**: si falta el dato, se aproxima **10 %** de la inercia de su polea.\n"
            "- **Impulsor** \(J_{imp}\): manuales **Metso**.\n\n"
            "Las inercias del lado bomba giran a $\\omega_p=\\omega_m/r$. "
            "Igualando energías cinéticas a una $\\omega_m$ común, los términos del lado de la bomba se dividen por $r^2$."
        )

st.markdown("---")

# =============================================================================
# 3) Tiempo inercial (par disponible vs rampa VDF)
# =============================================================================
st.markdown("## 3) Tiempo inercial (par disponible vs rampa VDF)")
rampa_vdf = st.slider("Rampa VDF en el motor [rpm/s]", min_value=10, max_value=600, value=100, step=5)

n_dot_torque = float("nan")
if not (math.isnan(J_eq) or math.isnan(T_nom_nm) or J_eq <= 0):
    n_dot_torque = (60.0 / (2.0 * math.pi)) * (T_nom_nm / J_eq)  # rpm/s
    t_par = (n_m_max - n_m_min) / max(n_dot_torque, 1e-9)
else:
    t_par = float("nan")

t_rampa = (n_m_max - n_m_min) / max(rampa_vdf, 1e-9)

cA, cB, cC = st.columns(3)
with cA:
    st.markdown(f"- Aceleración por par: {val_green(n_dot_torque, 'rpm/s')}", unsafe_allow_html=True)
with cB:
    st.markdown(f"- Tiempo por par (25→50 Hz): {val_green(t_par, 's')}", unsafe_allow_html=True)
with cC:
    st.markdown(f"- Tiempo por rampa VDF: {val_blue(t_rampa, 's')}", unsafe_allow_html=True)

def pill(text: str, bg: str = "#e8f5e9", color: str = "#1b5e20"):
    st.markdown(
        f"<div style='border-left:5px solid {color}; background:{bg}; padding:0.8rem 1rem; border-radius:0.5rem; margin-top:0.4rem'><b style='color:{color}'>{text}</b></div>",
        unsafe_allow_html=True,
    )

lim3 = "Tiempo limitante (sección 3): "
pill(lim3 + (f"**por par** = {fmt_num(t_par, 's')}" if (not math.isnan(t_par) and t_par > t_rampa) else f"**por rampa VDF** = {fmt_num(t_rampa, 's')}"))

with st.expander("Detalles y fórmulas — Sección 3", expanded=False):
    st.latex(r"\text{Hipótesis: } T_{\mathrm{disp}}=T_{\mathrm{nom}} \text{ (25–50 Hz), pérdidas mecánicas despreciables.}")
    st.latex(r"J_{eq}\,\dot\omega_m = T_{\mathrm{disp}}\ \Rightarrow\ \dot\omega_m = \frac{T_{\mathrm{disp}}}{J_{eq}}")
    st.latex(r"n_m=\frac{60}{2\pi}\,\omega_m\ \Rightarrow\ \dot n_m=\frac{60}{2\pi}\,\frac{T_{\mathrm{disp}}}{J_{eq}}")
    st.latex(r"t_{\mathrm{par}}=\frac{\Delta n_m}{\dot n_m},\qquad t_{\mathrm{rampa}}=\frac{\Delta n_m}{\text{rampa}}")

st.markdown("---")

# =============================================================================
# 4) Integración con carga hidráulica (slider limitado a 25–50 Hz)
# =============================================================================
st.markdown("## 4) Integración con carga hidráulica (25–50 Hz)")
st.latex(r"H(Q) = H_0 + K\,\left(\frac{Q}{3600}\right)^2\qquad [\,Q:\ \mathrm{m^3/h},\ H:\ \mathrm{m}\,]")
st.latex(r"\eta(Q)=\begin{cases}\eta_a+\eta_b\,\beta+\eta_c\,\beta^2,\quad \beta=Q/Q_{\mathrm{ref}}, & \text{si hay coeficientes}\\[4pt]\eta_\text{dataset}, & \text{en caso contrario}\end{cases},\ \ \eta\in[\eta_{\min},\eta_{\max}]")
st.latex(r"T_{\mathrm{pump}}=\frac{\rho g Q_s H(Q)}{\eta(Q)\,\omega_p},\ \ \omega_p=\frac{2\pi n_p}{60},\qquad T_{\mathrm{load,m}}=\frac{T_{\mathrm{pump}}}{r}")
st.latex(r"\Delta t = \frac{J_{eq}\,\Delta\omega_m}{T_{\mathrm{disp}}-T_{\mathrm{load,m}}},\qquad \frac{dt}{dn_m}=\frac{J_{eq}\,(2\pi/60)}{T_{\mathrm{disp}}-T_{\mathrm{load,m}}}")

# Límite del slider por TAG: caudales a 25 y 50 Hz
def q_at_speed(n_p: float) -> float:
    if not math.isnan(Q_ref) and not math.isnan(n_ref) and n_ref > 0:
        return Q_ref * (n_p / n_ref)
    if not math.isnan(Q_min_ds) and not math.isnan(Q_max_ds) and not math.isnan(n_p_min) and not math.isnan(n_p_max) and n_p_max > n_p_min:
        # Afinidad lineal entre extremos conocidos del dataset
        return Q_min_ds + (Q_max_ds - Q_min_ds) * (n_p - n_p_min) / (n_p_max - n_p_min)
    return float("nan")

Q25 = Q_min_ds if not math.isnan(Q_min_ds) else q_at_speed(n_p_min)
Q50 = Q_max_ds if not math.isnan(Q_max_ds) else q_at_speed(n_p_max)

# Asegurar orden y valores razonables
if Q25 > Q50:
    Q25, Q50 = Q50, Q25

# Slider EXCLUSIVAMENTE dentro [Q25, Q50] del TAG
Q_min, Q_max = st.slider(
    "Rango de caudal considerado [m³/h] (limitado a 25–50 Hz del TAG)",
    min_value=float(Q25), max_value=float(Q50),
    value=(float(Q25), float(Q50)),
    step=1.0,
)

# Mallas sobre el rango 25–50 Hz
N = 600
n_m_grid = np.linspace(n_m_min, n_m_max, N)
n_p_grid = np.linspace(n_p_min, n_p_max, N)

# Mapear Q a lo largo del rango rpm, acotado por el slider
Q_grid = Q_min + (Q_max - Q_min) * (n_p_grid - n_p_min) / max((n_p_max - n_p_min), 1e-9)
Q_s    = Q_grid / 3600.0

# Eficiencia
def eta_from_Q(Q_m3h: np.ndarray) -> np.ndarray:
    # Preferir polinomio si hay coeficientes; si no, valor fijo o 0.72
    if not (math.isnan(eta_a) or math.isnan(eta_b) or math.isnan(eta_c) or math.isnan(Q_ref) or Q_ref <= 0):
        beta = Q_m3h / Q_ref
        e = eta_a + eta_b * beta + eta_c * (beta**2)
    elif not math.isnan(eta_fixed):
        e = np.full_like(Q_m3h, float(eta_fixed))
    else:
        e = np.full_like(Q_m3h, 0.72)
    emin = 0.4 if math.isnan(eta_min) else eta_min
    emax = 0.88 if math.isnan(eta_max) else eta_max
    return np.clip(e, emin, emax)

eta_grid = eta_from_Q(Q_grid)
H_grid   = H0_m + K_m_s2 * (Q_s**2)
P_h_grid = rho *  g * Q_s * H_grid / np.maximum(eta_grid, 1e-6)  # [W]
omega_p  = 2.0 * math.pi * n_p_grid / 60.0
T_pump   = np.where(omega_p > 1e-9, P_h_grid / omega_p, 0.0)     # [N·m] (eje bomba)
T_load_m = T_pump / max(r_nm_np, 1e-9)                           # reflejo al motor
T_disp_m = np.full_like(T_load_m, T_nom_nm)

# Gráfico de par
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

# Gráfico de potencia hidráulica
fig_p = go.Figure()
fig_p.add_trace(go.Scatter(x=n_p_grid, y=P_h_grid/1000.0, mode="lines", name="Potencia hidráulica [kW]"))
fig_p.update_layout(
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis_title="Velocidad bomba n_p [rpm]",
    yaxis_title="Potencia hidráulica [kW]",
    legend=dict(orientation="h"),
    height=320,
)
st.plotly_chart(fig_p, use_container_width=True)

# Integración temporal con chequeo de suficiencia de par
T_net = T_disp_m - T_load_m
if np.any(T_net <= 0):
    st.warning("⚠️ Par motor insuficiente en parte del rango 25–50 Hz seleccionado. No se puede integrar el tiempo.")
    t_hyd = float("nan")
    # Aun así, mostrar dt/dn_m con valores válidos
    dt_dn = np.where(T_net > 0, J_eq * (2.0 * math.pi / 60.0) / T_net, np.nan)
else:
    omega_m = 2.0 * math.pi * n_m_grid / 60.0
    d_omega = np.diff(omega_m)
    dt = (J_eq * d_omega) / T_net[:-1]
    t_hyd = float(np.sum(dt))
    dt_dn = J_eq * (2.0 * math.pi / 60.0) / T_net

# Resultados + “pill” alineado debajo
c4a, c4b = st.columns([1, 2])
with c4a:
    st.markdown(f"- Tiempo por carga **hidráulica** (integración): {val_green(t_hyd, 's')}", unsafe_allow_html=True)
with c4b:
    base = "Tiempo limitante (sección 4): "
    if math.isnan(t_hyd):
        pill(base + f"**rampa VDF** = {fmt_num((n_m_max - n_m_min)/max(rampa_vdf,1e-9), 's')}")
    else:
        t_lim_4 = max(t_hyd, (n_m_max - n_m_min)/max(rampa_vdf,1e-9))
        which = "hidráulica" if t_hyd >= (n_m_max - n_m_min)/max(rampa_vdf,1e-9) else "rampa VDF"
        pill(f"{base}**{which} = {fmt_num(t_lim_4, 's')}**")

# Curva de integración dt/dn_m y área
fig_dt = go.Figure()
fig_dt.add_trace(go.Scatter(x=n_m_grid, y=dt_dn, name="dt/dn_m [s/rpm]", mode="lines"))
# Sombrear área válida
fig_dt.add_trace(
    go.Scatter(
        x=np.concatenate([n_m_grid, n_m_grid[::-1]]),
        y=np.concatenate([np.nan_to_num(dt_dn, nan=0.0), np.zeros_like(dt_dn)[::-1]]),
        fill="toself",
        name="Área integrada",
        opacity=0.2,
        line=dict(width=0),
        showlegend=True,
    )
)
fig_dt.update_layout(
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis_title="Velocidad motor n_m [rpm]",
    yaxis_title="dt/dn_m [s/rpm]",
    legend=dict(orientation="h"),
    height=320,
)
st.plotly_chart(fig_dt, use_container_width=True)

with st.expander("Detalles y fórmulas — Sección 4", expanded=False):
    st.markdown(
        r"""
- **Curva del sistema:** \(H(Q)=H_0+K\,(Q/3600)^2\).
- **Afinidad (25–50 Hz):** \(Q\propto n_p\). Con el *slider* restringimos \(Q\) a los caudales correspondientes a \(n_{p,\min}\) y \(n_{p,\max}\) del **TAG**.
- **Eficiencia:** si hay coeficientes, \(\eta(Q)=\eta_a+\eta_b\beta+\eta_c\beta^2\) con \(\beta=Q/Q_{\mathrm{ref}}\); en caso contrario se usa \(\eta\) del dataset. Se acota a \([\eta_{\min},\eta_{\max}]\).
- **Par resistente:** \(T_{\mathrm{pump}}=\dfrac{\rho g Q_s H(Q)}{\eta(Q)\,\omega_p}\), con \(Q_s=Q/3600\), \(\omega_p=2\pi n_p/60\); reflejo al motor \(T_{\mathrm{load,m}}=T_{\mathrm{pump}}/r\).
- **Integración temporal:** \(\Delta t = \dfrac{J_{eq}\,\Delta\omega_m}{T_{\mathrm{disp}}-T_{\mathrm{load,m}}}\). Integrando \(\dfrac{dt}{dn_m}=\dfrac{J_{eq}(2\pi/60)}{T_{\mathrm{disp}}-T_{\mathrm{load,m}}}\) entre \(n_{m,\min}\) y \(n_{m,\max}\) se obtiene el tiempo por carga hidráulica.
        """
    )

st.markdown("---")

# =============================================================================
# 5) Exportar CSV del TAG con resultados
# =============================================================================
st.markdown("## 5) Exportar CSV del TAG")
out = {
    "TAG": tag_sel,
    "pump_model": pump_model,
    "r_nm_np": r_nm_np,
    "n_m_min_rpm": n_m_min, "n_m_max_rpm": n_m_max,
    "n_p_min_rpm": n_p_min, "n_p_max_rpm": n_p_max,
    "J_m": J_m, "J_driver": J_driver, "J_bushing_driver": J_bdrv,
    "J_driven": J_driven, "J_bushing_driven": J_bdrn, "J_imp": J_imp,
    "J_eq": J_eq,
    "T_nom_nm": T_nom_nm,
    "rampa_vdf_rpm_s": rampa_vdf,
    "t_por_par_s": t_par,
    "t_por_rampa_s": (n_m_max - n_m_min)/max(rampa_vdf,1e-9),
    "H0_m": H0_m, "K_m_s2": K_m_s2,
    "Q25_m3h": Q25, "Q50_m3h": Q50,
    "Q_slider_min_m3h": Q_min, "Q_slider_max_m3h": Q_max,
    "rho_kgm3": rho,
    "eta_a": eta_a, "eta_b": eta_b, "eta_c": eta_c, "eta_fixed": eta_fixed,
    "t_hidraulica_s": t_hyd,
}
df_out = pd.DataFrame([out])
csv_bytes = df_out.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Descargar CSV del TAG (con resultados)", data=csv_bytes,
                   file_name=f"reporte_{tag_sel}.csv", mime="text/csv")
