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

# Colores para estilo
BLUE = "#1f77b4"   # Dado (dataset) → azul
GREEN = "#2ca02c"  # Calculado      → verde


# =============================================================================
# Utilidades
# =============================================================================
def dataset_path() -> Path:
    return Path(__file__).with_name("dataset.csv")


def images_path(name: str) -> Path:
    return Path(__file__).with_name("images") / name


def get_num(x) -> float:
    """Convierte a float tolerante a coma decimal y símbolos."""
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
    """Formatea números con coma decimal; cadenas se devuelven tal cual."""
    if isinstance(x, str):
        return f"{x} {unit}".strip()
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    s = f"{x:,.{ndigits}f}"
    # en- > es- (coma decimal)
    s = s.replace(",", "_").replace(".", ",").replace("_", ".")
    return f"{s} {unit}".strip()


def color_value(text: str, color: str = BLUE, bold: bool = True) -> str:
    w = "600" if bold else "400"
    return f'<span style="color:{color}; font-weight:{w}">{text}</span>'


def val_blue(x, unit: str = "", ndigits: int = 2) -> str:
    return color_value(fmt_num(x, unit, ndigits), BLUE)


def val_green(x, unit: str = "", ndigits: int = 2) -> str:
    return color_value(fmt_num(x, unit, ndigits), GREEN)


def pill(text: str, bg: str = "#e8f5e9", color: str = "#1b5e20"):
    st.markdown(
        f"""
        <div style="border-left: 5px solid {color}; background:{bg};
                    padding:0.8rem 1rem; border-radius:0.5rem; margin-top:0.5rem">
            <b style="color:{color}">{text}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Mapeo de columnas del dataset (acepta variantes en mayúsculas/minúsculas)
# =============================================================================
# Para cada clave, proveemos una lista de nombres de columnas candidatos.
ATTR = {
    # Identificación / rating
    "TAG":                 {"cols": ["TAG"],                        "unit": "",      "type": "str"},
    "pump_model":          {"cols": ["pumpmodel", "pump_model"],    "unit": "",      "type": "str"},
    "impeller_d_mm":       {"cols": ["impeller_d_mm"],              "unit": "mm",    "type": "num"},
    "motorpower_kw":       {"cols": ["motorpower_kw"],              "unit": "kW",    "type": "num"},
    "t_nom_nm":            {"cols": ["t_nom_nm"],                   "unit": "N·m",   "type": "num"},
    "r":                   {"cols": ["r_trans", "r_nm_np"],         "unit": "",      "type": "num"},
    "n_m_min":             {"cols": ["motor_n_min_rpm"],            "unit": "rpm",   "type": "num"},
    "n_m_max":             {"cols": ["motor_n_max_rpm"],            "unit": "rpm",   "type": "num"},
    "n_p_min":             {"cols": ["pump_n_min_rpm"],             "unit": "rpm",   "type": "num"},
    "n_p_max":             {"cols": ["pump_n_max_rpm"],             "unit": "rpm",   "type": "num"},

    # Inercias (catálogo TB Woods / Metso)
    "J_m":                 {"cols": ["motor_j_kgm2"],               "unit": "kg·m²", "type": "num"},
    "J_driver":            {"cols": ["driverpulley_j_kgm2"],        "unit": "kg·m²", "type": "num"},
    "J_sleeve_driver":     {"cols": ["driverbushing_j_kgm2"],       "unit": "kg·m²", "type": "num"},
    "J_driven":            {"cols": ["drivenpulley_j_kgm2",  # nuevo
                                     "drivenpulley_j_Kgm2"],        "unit": "kg·m²", "type": "num"},
    "J_sleeve_driven":     {"cols": ["drivenbushing_j_kgm2", # nuevo
                                     "drivenbushing_j_Kgm2"],       "unit": "kg·m²", "type": "num"},
    "J_imp":               {"cols": ["impeller_j_kgm2"],            "unit": "kg·m²", "type": "num"},

    # Hidráulica / pulpa
    "H0_m":                {"cols": ["H0_m", "h0_m"],               "unit": "m",     "type": "num"},
    "K_H_per_Q":           {"cols": ["k_h_per_q", "K_H_per_Q"],     "unit": "—",     "type": "num"},  # H=H0+K*(Q/3600)^2
    "Qmin_m3h":            {"cols": ["Qmin_m3h", "qmin_m3h"],       "unit": "m³/h",  "type": "num"},
    "Qbest_m3h":           {"cols": ["Qbest_m3h", "qbest_m3h"],     "unit": "m³/h",  "type": "num"},
    "Qmax_m3h":            {"cols": ["Qmax_m3h", "qmax_m3h"],       "unit": "m³/h",  "type": "num"},
    "eta":                 {"cols": ["eta", "efficiency"],          "unit": "",      "type": "num"},
    "SlurryDensity":       {"cols": ["slurrydensity_kgm3", "SlurryDensity_Kgm3"],
                                                                  "unit": "kg/m³", "type": "num"},
}


def _get_from_row(row: pd.Series, candidates: list[str], default=np.nan):
    """Busca el primer nombre de columna presente (case-insensitive)."""
    idx = list(row.index)
    lower_idx = {c.lower(): c for c in idx}
    for name in candidates:
        if name in row:
            return row[name]
        lname = name.lower()
        if lname in lower_idx:
            return row[lower_idx[lname]]
    return default


def get_attr(row: pd.Series, key: str, default=np.nan):
    """Obtiene valor de una clave del mapeo ATTR para una fila del df."""
    meta = ATTR[key]
    v = _get_from_row(row, meta["cols"], default)
    if meta["type"] == "num":
        return get_num(v)
    # string
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return str(v)


# =============================================================================
# Cargar dataset + UI (logos y título)
# =============================================================================
# Lectura del CSV (siempre presente)
try:
    df_raw = pd.read_csv(dataset_path(), sep=";", decimal=",")
    df_raw.columns = [c.strip() for c in df_raw.columns]
except Exception as e:
    st.error(f"No se pudo leer dataset.csv: {e}")
    st.stop()

# Encabezado con logos + título
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

# Selector de TAG
if "TAG" not in df_raw.columns:
    st.error("El dataset debe contener la columna 'TAG'.")
    st.stop()

tags = df_raw["TAG"].astype(str).tolist()
tag_sel = st.sidebar.selectbox("Selecciona TAG", tags, index=0)
row = df_raw[df_raw["TAG"].astype(str) == str(tag_sel)].iloc[0]


# =============================================================================
# 1) Parámetros
# =============================================================================
st.markdown("## 1) Parámetros")

# Lectura de atributos base
txt_pump_model = get_attr(row, "pump_model")
P_motor_kW     = get_attr(row, "motorpower_kw")
T_nom_nm       = get_attr(row, "t_nom_nm")
r_nm_np        = get_attr(row, "r")
D_imp_mm       = get_attr(row, "impeller_d_mm")

n_m_min = get_attr(row, "n_m_min")
n_m_max = get_attr(row, "n_m_max")
n_p_min = get_attr(row, "n_p_min")
n_p_max = get_attr(row, "n_p_max")

# Hidráulica
H0_m     = get_attr(row, "H0_m")
K_coef   = get_attr(row, "K_H_per_Q")
eta_ds   = get_attr(row, "eta")
Q_min_ds = get_attr(row, "Qmin_m3h")
Q_best   = get_attr(row, "Qbest_m3h")   # no se muestra; lo conservamos por si se usa en el futuro
Q_max_ds = get_attr(row, "Qmax_m3h")
rho      = get_attr(row, "SlurryDensity")
if np.isnan(rho) or rho <= 0:
    rho = 1000.0  # fallback agua

# Mostrar
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
    st.markdown(
        f"- Velocidad motor min–max: {val_blue(n_m_min, 'rpm', 0)} – {val_blue(n_m_max, 'rpm', 0)}",
        unsafe_allow_html=True,
    )

with c3:
    st.markdown("**Bomba (25–50 Hz por afinidad)**")
    st.markdown(
        f"- Velocidad bomba min–max: {val_blue(n_p_min, 'rpm', 0)} – {val_blue(n_p_max, 'rpm', 0)}",
        unsafe_allow_html=True,
    )
    st.markdown(f"- Densidad de pulpa ρ: {val_blue(rho, 'kg/m³', 0)}", unsafe_allow_html=True)

st.markdown("---")


# =============================================================================
# 2) Cálculo de inercia equivalente
# =============================================================================
st.header("2) Cálculo de inercia equivalente")

colL, colR = st.columns([1.1, 1])

with colL:
    st.subheader("Inercias individuales")

    J_m = get_attr(row, "J_m")
    J_driver = get_attr(row, "J_driver")
    J_sleeve_driver = get_attr(row, "J_sleeve_driver")
    if np.isnan(J_sleeve_driver):  # si falta: 10% de la polea motriz
        J_sleeve_driver = 0.10 * (J_driver if not np.isnan(J_driver) else 0.0)

    J_driven = get_attr(row, "J_driven")
    J_sleeve_driven = get_attr(row, "J_sleeve_driven")
    if np.isnan(J_sleeve_driven):  # si falta: 10% de la polea conducida
        J_sleeve_driven = 0.10 * (J_driven if not np.isnan(J_driven) else 0.0)

    J_imp = get_attr(row, "J_imp")
    r     = max(get_attr(row, "r"), 1e-6)

    st.markdown(f"- Motor (J_m): {val_blue(J_m, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Polea motriz (J_driver): {val_blue(J_driver, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Manguito motriz (J_sleeve_driver≈10% J_driver): {val_blue(J_sleeve_driver, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Polea conducida (J_driven): {val_blue(J_driven, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Manguito conducido (J_sleeve_driven≈10% J_driven): {val_blue(J_sleeve_driven, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Impulsor/rotor de bomba (J_imp): {val_blue(J_imp, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Relación r (n_m/n_p): {val_blue(r, '')}", unsafe_allow_html=True)

    # J_eq en el eje del motor (inercias del lado bomba vistas / r²)
    J_eq = (J_m + J_driver + J_sleeve_driver) + (J_driven + J_sleeve_driven + J_imp) / (r**2)
    st.markdown(f"**Inercia equivalente (J_eq):** {val_green(J_eq, 'kg·m²')}", unsafe_allow_html=True)

with colR:
    st.subheader("Fórmula utilizada")
    st.latex(
        r"J_{\mathrm{eq}} \;=\; J_m \;+\; J_{\mathrm{driver}} \;+\; J_{\mathrm{sleeve,driver}} \;+\; \frac{J_{\mathrm{driven}} + J_{\mathrm{sleeve,driven}} + J_{\mathrm{imp}}}{r^2}"
    )
    with st.expander("Formulación de las inercias por componente", expanded=True):
        st.markdown(
            "- **Poleas** (J_driver, J_driven): obtenidas de catálogo **TB Woods**.\n"
            "- **Manguitos** (J_sleeve_driver, J_sleeve_driven): si no hay dato, se aproxima **10%** de la inercia de su polea.\n"
            "- **Impulsor** (J_imp): de manuales **Metso**.\n\n"
            "Las inercias del lado bomba giran a \(\\omega_p=\\omega_m/r\\). "
            "Igualando energías cinéticas a una \(\\omega_m\\) común se obtiene la división por \(r^2\) para términos del lado de la bomba."
        )

st.markdown("---")


# =============================================================================
# 3) Tiempo inercial (par disponible vs rampa)
# =============================================================================
st.markdown("## 3) Tiempo inercial (par disponible vs rampa VDF)")

# Slider rampa en el motor
rampa_vdf = st.slider("Rampa VDF en el motor [rpm/s]", min_value=10, max_value=400, value=100, step=5)

# Aceleración por par disponible (en rpm/s) y tiempos
if not (np.isnan(J_eq) or np.isnan(T_nom_nm) or J_eq <= 0):
    # \dot{ω}_m = T_disp / J_eq  → \dot{n}_m = (60/2π) * (T_disp/J_eq)
    n_dot_torque = (60.0 / (2.0 * math.pi)) * (T_nom_nm / J_eq)  # rpm/s
    t_par = (n_m_max - n_m_min) / max(n_dot_torque, 1e-9)        # s
else:
    n_dot_torque = float("nan")
    t_par = float("nan")

t_rampa = (n_m_max - n_m_min) / max(rampa_vdf, 1e-9)

cA, cB, cC = st.columns(3)
with cA:
    st.markdown(f"- Aceleración por par (lado motor): {val_green(n_dot_torque, 'rpm/s')}", unsafe_allow_html=True)
with cB:
    st.markdown(f"- Tiempo por par (25→50 Hz): {val_green(t_par, 's')}", unsafe_allow_html=True)
with cC:
    st.markdown(f"- Tiempo por rampa VDF: {val_blue(t_rampa, 's')}", unsafe_allow_html=True)

lim3_txt = "Tiempo limitante (sección 3): "
if not np.isnan(t_par) and t_par > t_rampa:
    pill(lim3_txt + f"**por par** = {fmt_num(t_par, 's')}")
    limit3_name, limit3_val = "par", t_par
else:
    pill(lim3_txt + f"**por rampa VDF** = {fmt_num(t_rampa, 's')}")
    limit3_name, limit3_val = "rampa VDF", t_rampa

with st.expander("Detalles y fórmulas — Sección 3", expanded=False):
    st.markdown(
        "- **Hipótesis:** par del motor constante e igual al nominal \(T_{disp}=T_{nom}\) en 25–50 Hz; "
        "las pérdidas mecánicas son despreciables.\n"
        "- **Dinámica rotacional (eje motor):**\n"
        "  \\[ J_{eq}\\,\\dot\\omega_m = T_{disp} \\;\\Rightarrow\\; \\dot\\omega_m = \\tfrac{T_{disp}}{J_{eq}}. \\]\n"
        "- **Conversión a rpm:**  \(n_m=\\tfrac{60}{2\\pi}\\,\\omega_m\\)  ⇒\n"
        "  \\[ \\dot n_m = \\tfrac{60}{2\\pi}\\,\\tfrac{T_{disp}}{J_{eq}}. \\]\n"
        "- **Tiempo por par:**  \(t_{par}=\\dfrac{\\Delta n_m}{\\dot n_m}\\).\n"
        "- **Tiempo por rampa VDF:**  \(t_{rampa}=\\dfrac{\\Delta n_m}{\\text{rampa}}\).\n"
        "- **Criterio:** se toma como **limitante** el mayor entre \(t_{par}\) y \(t_{rampa}\)."
    )

st.markdown("---")


# =============================================================================
# 4) Integración con carga hidráulica
# =============================================================================
st.markdown("## 4) Integración con carga hidráulica")
st.latex(r"H(Q)=H_0+K\left(\dfrac{Q}{3600}\right)^2 \qquad \big[Q:\ \mathrm{m^3/h},\ H:\ \mathrm{m}\big]")

c4a, c4b, c4c = st.columns(3)
with c4a:
    st.markdown(f"- H₀: {val_blue(H0_m, 'm')}", unsafe_allow_html=True)
with c4b:
    st.markdown(f"- Coeficiente K: {val_blue(K_coef, '', 3)}", unsafe_allow_html=True)
with c4c:
    st.markdown(f"- ρ: {val_blue(rho, 'kg/m³', 0)}", unsafe_allow_html=True)

# ----- Rango de caudales para el análisis (limitado a ±30%) -----
def _fallback_qmin_qmax():
    if not np.isnan(Q_min_ds) and not np.isnan(Q_max_ds) and Q_min_ds < Q_max_ds:
        return float(Q_min_ds), float(Q_max_ds)
    # fallback si faltan datos
    return 100.0, 300.0

qmin_base, qmax_base = _fallback_qmin_qmax()
qmin_allowed = 0.70 * qmin_base
qmax_allowed = 1.30 * qmax_base

Q_min, Q_max = st.slider(
    "Rango de caudal considerado [m³/h]  (límite: −30% desde Q_min y +30% desde Q_max)",
    min_value=float(qmin_allowed),
    max_value=float(qmax_allowed),
    value=(float(qmin_base), float(qmax_base)),
    step=1.0,
)

# Mapear velocidad bomba → caudal (lineal 25–50 Hz)
def Q_from_np(n_p: np.ndarray) -> np.ndarray:
    if n_p_max <= n_p_min + 1e-9:
        return np.full_like(n_p, Q_min)
    return Q_min + (Q_max - Q_min) * (n_p - n_p_min) / (n_p_max - n_p_min)

# Eficiencia: constante del dataset con clipping básico
def eta_from_Q(Q_m3h: np.ndarray) -> np.ndarray:
    eta0 = 0.72 if np.isnan(eta_ds) else float(eta_ds)
    return np.full_like(Q_m3h, np.clip(eta0, 0.4, 0.88))

# Mallas en velocidad del motor/bomba
N = 600
n_m_grid = np.linspace(n_m_min, n_m_max, N)
n_p_grid = n_m_grid / max(r_nm_np, 1e-9)

Q_grid   = Q_from_np(n_p_grid)          # m³/h
q_grid   = Q_grid / 3600.0              # m³/s
H_grid   = H0_m + K_coef * (q_grid**2)  # m
eta_grid = eta_from_Q(Q_grid)           # —

# Potencias y pares
g = 9.81
P_h_grid = rho * g * q_grid * H_grid / np.maximum(eta_grid, 1e-6)   # W
omega_p  = 2.0 * math.pi * n_p_grid / 60.0                          # rad/s
T_pump   = np.where(omega_p > 1e-9, P_h_grid / omega_p, 0.0)        # N·m (eje bomba)
T_load_m = T_pump / max(r_nm_np, 1e-9)                              # N·m (eje motor)
T_disp_m = np.full_like(T_load_m, T_nom_nm)                         # N·m (motor)

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

# Integración temporal con carga hidráulica
if not (np.isnan(J_eq) or J_eq <= 0):
    omega_m_grid = 2.0 * math.pi * n_m_grid / 60.0
    d_omega = np.diff(omega_m_grid)                         # rad/s
    T_net = T_disp_m - T_load_m                             # N·m
    T_net_clip = np.maximum(T_net, 1e-6)                    # evitar división por 0
    dt = (J_eq * d_omega) / T_net_clip[:-1]                 # s
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

with st.expander("Detalles y fórmulas — Sección 4", expanded=False):
    st.markdown(
        "- **Curva del sistema** (aprox. cuadrática en caudal):\n"
        "  \\[ H(Q)=H_0+K\\,(Q/3600)^2. \\]\n"
        "- **Afinidad** (25–50 Hz): \(Q\\propto n_p\\). Se interpola linealmente entre \(n_{p,\\min}\) y \(n_{p,\\max}\) para mapear \(n_p\\to Q\\).\n"
        "- **Eficiencia**: se usa una eficiencia global constante del dataset (acotada en \\([0{,}40,\\,0{,}88]\\)).\n"
        "- **Potencia hidráulica** (eje bomba):\n"
        "  \\[ P_h = \\frac{\\rho g\\,Q_s\\,H(Q)}{\\eta(Q)},\\qquad Q_s=Q/3600. \\]\n"
        "- **Par de bomba y reflejo al motor**:\n"
        "  \\[ T_{pump} = \\frac{P_h}{\\omega_p},\\qquad \\omega_p=\\frac{2\\pi n_p}{60},\\qquad T_{load,m}=\\frac{T_{pump}}{r}. \\]\n"
        "- **Integración temporal** en el eje del motor (par neto variable):\n"
        "  \\[ J_{eq}\\,\\dot\\omega_m = T_{disp}-T_{load,m}(n) \\;\\Rightarrow\\; "
        "     \\Delta t = \\frac{J_{eq}\\,\\Delta\\omega_m}{T_{disp}-T_{load,m}}. \\]\n"
        "  Se integra numéricamente entre \(n_{m,\\min}\\) y \(n_{m,\\max}\\).\n"
        "- **Criterio**: se reporta el tiempo por hidráulica y se compara con el tiempo por rampa; el mayor es el **limitante**."
    )

st.markdown("---")


# =============================================================================
# 5) Exportar resultados (CSV)
# =============================================================================
st.markdown("## 5) Exportar resultados (CSV)")

# Valores finales usados (incluye fallback del 10% en manguitos)
J_sleeve_driver_used = J_sleeve_driver
J_sleeve_driven_used = J_sleeve_driven

# Para exportación – determinar tiempo limitante global
def _safe(v):  # para manejar NaN en el 'max'
    return -np.inf if np.isnan(v) else v

candidates = [("par", t_par), ("rampa VDF", t_rampa), ("hidráulica", t_hyd)]
limit_global_name, limit_global_val = max(candidates, key=lambda kv: _safe(kv[1]))

summary = {
    # Identificación
    "TAG": tag_sel,
    "pump_model": txt_pump_model,
    "impeller_d_mm": D_imp_mm,

    # Motor & transmisión
    "motorpower_kw": P_motor_kW,
    "t_nom_nm": T_nom_nm,
    "r_nm_np": r_nm_np,
    "n_m_min_rpm": n_m_min,
    "n_m_max_rpm": n_m_max,
    "n_p_min_rpm": n_p_min,
    "n_p_max_rpm": n_p_max,

    # Inercias
    "J_m_kgm2": J_m,
    "J_driver_kgm2": J_driver,
    "J_sleeve_driver_used_kgm2": J_sleeve_driver_used,
    "J_driven_kgm2": J_driven,
    "J_sleeve_driven_used_kgm2": J_sleeve_driven_used,
    "J_imp_kgm2": J_imp,
    "J_eq_kgm2": J_eq,

    # Hidráulica base
    "H0_m": H0_m,
    "K_H_per_Q": K_coef,
    "rho_kgm3": rho,
    "Q_slider_min_m3h": Q_min,
    "Q_slider_max_m3h": Q_max,
    "eta_used": 0.72 if np.isnan(eta_ds) else float(eta_ds),

    # Rampa y tiempos
    "rampa_vdf_rpmps": rampa_vdf,
    "n_dot_torque_rpms": n_dot_torque,
    "t_par_s": t_par,
    "t_rampa_s": t_rampa,
    "t_hidraulica_s": t_hyd,
    "tiempo_limitante_global": limit_global_name,
    "tiempo_limitante_valor_s": limit_global_val,
}

df_out = pd.DataFrame([summary])
csv_bytes = df_out.to_csv(sep=";", index=False, decimal=",").encode("utf-8")
st.download_button(
    "⬇️ Descargar CSV (resumen del TAG)",
    data=csv_bytes,
    file_name=f"reporte_{tag_sel}_rampa_{int(rampa_vdf)}rpmps.csv",
    mime="text/csv",
)
