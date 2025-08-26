# app.py
# ─────────────────────────────────────────────────────────────────────────────
# Dashboard: Tiempo de reacción de bombas con VDF
# Secciones:
#   1) Parámetros (datos dados por TAG)
#   2) Cálculo de inercia equivalente (con detalle de formulación)
#   3) Tiempos "por par" vs "por rampa" + recuadro verde con tiempo limitante
#   4) Integración con carga hidráulica (curva del sistema, potencia eje y tiempo)
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

# Paleta para valores
BLUE = "#1f77b4"   # Dado (dataset)  → azul
GREEN = "#2ca02c"  # Calculado       → verde
GRAY = "#6c757d"

# Utilidades ------------------------------------------------------------------

def dataset_path() -> Path:
    """Ruta del dataset.csv (siempre en el repo junto a app.py)."""
    return Path(__file__).with_name("dataset.csv")

def images_path(name: str) -> Path:
    """Ruta a /images/<name> relativa al repo."""
    return Path(__file__).with_name("images") / name

def get_num(x) -> float:
    """Convierte a float números que pueden venir con separador decimal ',' y separadores de miles.
    Retorna NaN si no puede parsear.
    """
    if x is None:
        return float("nan")
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace(" ", "").replace("\u00a0", "")
    # normalizar decimal coma → punto
    s = s.replace(",", ".")
    # quitar cualquier char no numérico típico
    import re
    s = re.sub(r"[^0-9eE\+\-\.]", "", s)
    try:
        return float(s)
    except Exception:
        return float("nan")

def fmt_num(x, unit: str = "", ndigits: int = 2) -> str:
    """Formatea número con coma decimal (estilo ES)."""
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

def latex_center(s: str):
    st.markdown(
        f"<div style='text-align:center; margin: 0.5rem 0'>{s}</div>",
        unsafe_allow_html=True,
    )

# Mapeo de columnas (alias tolerantes) ---------------------------------------

ALIASES = {
    "tag": ["tag", "TAG", "Tag"],
    "pump_model": ["pumpmodel", "pump_model", "Modelo", "ModeloBomba"],
    "motor_power_kw": ["motorpower_kw", "motor_power_kw", "P_motor_kW"],
    "t_nom_nm": ["t_nom_nm", "torque_nom_nm", "T_nom_nm"],
    "r_nm_np": ["r_trans", "ratio_nm_np", "r", "r_nm_np"],
    # Inercias (catálogo TB Woods / manuales)
    "J_m": ["motor_j_kgm2", "J_motor_kgm2", "Jm_kgm2"],
    "J_driver": ["driver_j_kgm2", "J_driver_kgm2", "J_polea_motriz_kgm2"],
    "J_driven": ["driven_j_kgm2", "J_driven_kgm2", "J_polea_conducida_kgm2"],
    "J_imp": ["impeller_j_kgm2", "J_imp_kgm2", "J_rotor_bomba_kgm2"],
    "impeller_d_mm": ["impeller_d_mm", "D_imp_mm", "diametro_impulsor_mm"],
    # Velocidades de referencia (25–50 Hz por afinidad)
    "n_ref_rpm": ["n_ref_rpm", "n_p_ref_rpm", "n_bomba_ref_rpm"],
    "Q_ref_m3h": ["Q_ref_m3h", "Qref_m3h"],
    "Q_min_m3h": ["Q_min_m3h"],
    "Q_max_m3h": ["Q_max_m3h"],
    # Curva de sistema
    "H0_m": ["H0_m", "H_0_m"],
    "K_m_s2": ["K_m_s2", "K_m_per_(m3h)^2", "K_m"],
    # Eficiencia (polinomio cuadrático en beta = Q/Q_ref)
    "eta_a": ["eta_a"],
    "eta_b": ["eta_b"],
    "eta_c": ["eta_c"],
    "eta_beta": ["eta_beta"],  # si viene, usarla directamente
    "eta_min_clip": ["eta_min_clip"],
    "eta_max_clip": ["eta_max_clip"],
    # Densidad de pulpa
    "rho_kgm3": ["SlurryDensity_Kgm3", "rho_kgm3", "density_kgm3"],
}

def col_lookup(df: pd.DataFrame, key: str) -> str | None:
    """Devuelve el nombre de columna real para una clave canónica."""
    if key not in ALIASES:
        return None
    candidates = [c.lower() for c in df.columns]
    for alias in ALIASES[key]:
        if alias.lower() in candidates:
            return df.columns[candidates.index(alias.lower())]
    return None

def load_data() -> pd.DataFrame:
    path = dataset_path()
    df = pd.read_csv(path, sep=";", decimal=",", dtype=str)
    # normalizar espacios en encabezados
    df.columns = [c.strip() for c in df.columns]
    return df

# =============================================================================
# Encabezado con logos + título
# =============================================================================

colL, colC, colR = st.columns([1, 3, 1])
with colL:
    logo_metso = images_path("metso_logo.png")
    if logo_metso.exists():
        st.image(str(logo_metso), use_container_width=True)
with colC:
    st.markdown(
        "<h1 style='text-align:center; margin-top: 0.2rem'>Tiempo de reacción de bombas con VDF</h1>",
        unsafe_allow_html=True,
    )
with colR:
    logo_ausenco = images_path("ausenco_logo.png")
    if logo_ausenco.exists():
        st.image(str(logo_ausenco), use_container_width=True)

st.markdown("---")

# =============================================================================
# Carga del dataset + selector de TAG
# =============================================================================

df_raw = load_data()

tag_col = col_lookup(df_raw, "tag") or "TAG"
tags = df_raw[tag_col].astype(str).tolist()
tag_sel = st.sidebar.selectbox("Selecciona TAG", tags, index=0)

row = df_raw[df_raw[tag_col].astype(str) == str(tag_sel)].iloc[0]

# Extraer valores con alias
def get_val(key: str, default=np.nan) -> float:
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

# Rango de velocidades (25–50 Hz por afinidad usando n_ref si está disponible)
n_ref_rpm  = get_val("n_ref_rpm")
if np.isnan(n_ref_rpm):
    # fallback prudente
    n_ref_rpm = 1000.0
n_p_min = 0.5 * n_ref_rpm
n_p_max = 1.0 * n_ref_rpm
if not np.isnan(r_nm_np) and r_nm_np > 0:
    n_m_min = n_p_min * r_nm_np
    n_m_max = n_p_max * r_nm_np
else:
    n_m_min = 2 * n_p_min
    n_m_max = 2 * n_p_max

# Curva de sistema / eficiencia / densidad
H0_m     = get_val("H0_m")
K_m_s2   = get_val("K_m_s2")              # para H(Q)=H0 + K*(Q/3600)^2
eta_a    = get_val("eta_a")
eta_b    = get_val("eta_b")
eta_c    = get_val("eta_c")
eta_min  = get_val("eta_min_clip", 0.4)
eta_max  = get_val("eta_max_clip", 0.88)
eta_beta = get_val("eta_beta")            # si viene ya la eta en [0..1] a Q_ref

Q_ref    = get_val("Q_ref_m3h")
Q_min_ds = get_val("Q_min_m3h")
Q_max_ds = get_val("Q_max_m3h")

rho = get_val("rho_kgm3")
if np.isnan(rho) or rho <= 0:
    rho = 1000.0  # fallback agua

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

# =============================================================================
# 2) Cálculo de inercia equivalente
# =============================================================================

st.markdown("## 2) Cálculo de inercia equivalente")

left, right = st.columns([1.1, 1.2])
with left:
    st.markdown("**Inercias individuales**")
    # Manguitos: 10% de inercia de la polea correspondiente (aprox.)
    J_sleeve_driver = 0.10 * J_driver if not np.isnan(J_driver) else np.nan
    J_sleeve_driven = 0.10 * J_driven if not np.isnan(J_driven) else np.nan

    st.markdown(f"- Motor (J_m): {val_blue(J_m, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Polea motriz (J_driver): {val_blue(J_driver, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f\"- Manguito motriz (J_sleeve_driver≈10% J_driver): {val_green(J_sleeve_driver, 'kg·m²')}\", unsafe_allow_html=True)
    st.markdown(f"- Polea conducida (J_driven): {val_blue(J_driven, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f\"- Manguito conducido (J_sleeve_driven≈10% J_driven): {val_green(J_sleeve_driven, 'kg·m²')}\", unsafe_allow_html=True)
    st.markdown(f"- Impulsor/rotor de bomba (J_imp): {val_blue(J_imp, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Relación r (n_m/n_p): {val_blue(r_nm_np)}", unsafe_allow_html=True)

with right:
    st.markdown("**Fórmula utilizada**")
    st.latex(r"J_{\mathrm{eq}} \;=\; J_m \;+\; (J_{\mathrm{driver}} + J_{\mathrm{sleeve,drv}}) \;+\; \dfrac{J_{\mathrm{driven}} + J_{\mathrm{sleeve,drn}} + J_{\mathrm{imp}}}{r^2}")
    st.caption(
        "Las inercias del lado bomba giran a menor velocidad (ω_p = ω_m/r). "
        "Igualando energías cinéticas a una ω_m común, los términos del lado bomba se dividen por r²."
    )

# Inercia equivalente
if not np.isnan(r_nm_np) and r_nm_np > 0:
    J_eq = (
        (J_m if not np.isnan(J_m) else 0.0)
        + (J_driver if not np.isnan(J_driver) else 0.0)
        + (J_sleeve_driver if not np.isnan(J_sleeve_driver) else 0.0)
        + ( (J_driven if not np.isnan(J_driven) else 0.0)
            + (J_sleeve_driven if not np.isnan(J_sleeve_driven) else 0.0)
            + (J_imp if not np.isnan(J_imp) else 0.0)
          ) / (r_nm_np**2)
    )
else:
    J_eq = float("nan")

st.markdown(f"**Inercia equivalente (J_eq):** {val_green(J_eq, 'kg·m²')}", unsafe_allow_html=True)

with st.expander("Formulación de las inercias por componente"):
    st.markdown(
        "- **Poleas (J_driver, J_driven):** valores **obtenidos del catálogo TB Woods** para la serie y diámetro declarados.\n"
        "- **Manguitos/bushings:** se **aproximan** como el **10% de la inercia** de la polea correspondiente.\n"
        "- **Impulsor (J_imp):** tomado de **manuales Metso** para el modelo de bomba.\n"
        "- **Motor (J_m):** inercia del rotor informada por el fabricante del motor.\n"
    )

st.markdown("---")

# =============================================================================
# 3) Tiempo inercial básico (por par vs por rampa)
# =============================================================================

st.markdown("## 3) Tiempo inercial (par disponible vs rampa VDF)")

# Rampa VDF (rpm/s) en el lado motor
rampa_vdf = st.slider("Rampa VDF en el motor [rpm/s]", min_value=10, max_value=400, value=100, step=5)

# Tiempo por par (sin hidráulica):  ṅ = (60/2π) * T / J_eq  → tiempo = Δn / ṅ
if not (np.isnan(J_eq) or np.isnan(T_nom_nm) or J_eq <= 0):
    n_dot_torque = (60.0 / (2.0 * math.pi)) * (T_nom_nm / J_eq)  # rpm/s
    t_par = (n_m_max - n_m_min) / max(n_dot_torque, 1e-9)        # s
else:
    t_par = float("nan")

# Tiempo por rampa (control):  Δn / rampa
t_rampa = (n_m_max - n_m_min) / max(rampa_vdf, 1e-9)

# Mostrar
cA, cB, cC = st.columns(3)
with cA:
    st.markdown(f"- Aceleración por par (lado motor): {val_green(n_dot_torque, 'rpm/s')}", unsafe_allow_html=True)
with cB:
    st.markdown(f"- Tiempo por par (25→50 Hz): {val_green(t_par, 's')}", unsafe_allow_html=True)
with cC:
    st.markdown(f"- Tiempo por rampa VDF: {val_blue(t_rampa, 's')}", unsafe_allow_html=True)

# Recuadro: tiempo limitante (sección 3, solo comparando par vs rampa)
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

# Fórmula de la curva (modelo cuadrático por defecto con K_m_s2)
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

# Rango de caudal asociado a 25–50 Hz (si vienen Q_min/Q_max los usamos, sino proponemos)
if np.isnan(Q_min_ds) or np.isnan(Q_max_ds) or Q_min_ds >= Q_max_ds:
    # Proponer en torno a Q_ref
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
    min_value=0.0, max_value=max(5000.0, qmax_default*1.2),
    value=(float(qmin_default), float(qmax_default)),
    step=1.0,
)

# Afinidad: asociamos Q_min↔n_p_min y Q_max↔n_p_max (relación lineal)
def Q_from_np(n_p: np.ndarray) -> np.ndarray:
    if n_p_max <= n_p_min + 1e-9:
        return np.full_like(n_p, Q_min)
    return Q_min + (Q_max - Q_min) * (n_p - n_p_min) / (n_p_max - n_p_min)

# Eficiencia bomba vs Q
def eta_from_Q(Q_m3h: np.ndarray) -> np.ndarray:
    if not np.isnan(eta_beta) and eta_beta > 0 and not np.isnan(eta_min) and not np.isnan(eta_max):
        # Si η_beta viene ya como valor, lo usamos constante (rareza en algunos datasets)
        e = np.full_like(Q_m3h, fill_value=float(eta_beta))
    elif not (np.isnan(eta_a) or np.isnan(eta_b) or np.isnan(eta_c) or np.isnan(Q_ref) or Q_ref <= 0):
        beta = Q_m3h / Q_ref
        e = eta_a + eta_b * beta + eta_c * (beta**2)
    else:
        e = np.full_like(Q_m3h, 0.72)  # fallback
    # clip
    emin = eta_min if not np.isnan(eta_min) else 0.4
    emax = eta_max if not np.isnan(eta_max) else 0.88
    return np.clip(e, emin, emax)

# Discretización sobre rpm motor (o bomba) para integrar tiempo real con carga
N = 600
n_m_grid = np.linspace(n_m_min, n_m_max, N)
n_p_grid = n_m_grid / max(r_nm_np, 1e-9)
Q_grid   = Q_from_np(n_p_grid)             # m³/h
q_grid   = Q_grid / 3600.0                 # m³/s
H_grid   = H0_m + K_m_s2 * (q_grid**2)     # m
eta_grid = eta_from_Q(Q_grid)

# Potencia hidráulica y torque de la bomba
P_h_grid = rho * g * q_grid * H_grid / np.maximum(eta_grid, 1e-6)   # W
omega_p  = 2.0 * math.pi * n_p_grid / 60.0                          # rad/s
T_pump   = np.where(omega_p > 1e-9, P_h_grid / omega_p, 0.0)        # N·m (eje bomba)

# Par resistente reflejado al motor (sin pérdidas en transmisión)
T_load_m = T_pump / max(r_nm_np, 1e-9)                              # N·m (eje motor)
T_disp_m = np.full_like(T_load_m, T_nom_nm)                         # par disponible motor (constante)

# Graficar par carga vs disponible
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
fig_p.add_trace(go.Scatter(x=n_p_grid, y=P_h_grid/1000.0, mode="lines", name="P hidráulica [kW]"))
fig_p.update_layout(
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis_title="Velocidad bomba n_p [rpm]",
    yaxis_title="Potencia hidráulica [kW]",
    legend=dict(orientation="h"),
    height=320,
)
st.plotly_chart(fig_p, use_container_width=True)

# Integración del tiempo real con carga: dω/dt = (T_disp - T_load_m)/J_eq  (lado motor)
if not (np.isnan(J_eq) or J_eq <= 0):
    omega_m_grid = 2.0 * math.pi * n_m_grid / 60.0  # rad/s
    d_omega = np.diff(omega_m_grid)

    # T neto
    T_net = T_disp_m - T_load_m
    # Si en algún punto T_net ≤ 0 → no acelera (división por 0). Lo tratamos como muy lento.
    T_net_clip = np.maximum(T_net, 1e-6)

    dt = (J_eq * d_omega) / T_net_clip[:-1]  # s
    t_hyd = float(np.sum(dt))
else:
    t_hyd = float("nan")

c4_1, c4_2 = st.columns(2)
with c4_1:
    st.markdown(f"- Tiempo por carga **hidráulica** (integración): {val_green(t_hyd, 's')}", unsafe_allow_html=True)
with c4_2:
    # Comparamos con la rampa de control
    t_lim_4 = max(t_hyd, t_rampa) if not np.isnan(t_hyd) else t_rampa
    which = "hidráulica" if (not np.isnan(t_hyd) and t_hyd > t_rampa) else "rampa VDF"
    pill(f"Tiempo limitante (sección 4): **{which} = {fmt_num(t_lim_4, 's')}**")

with st.expander("Detalles de la formulación (sección 4)"):
    st.markdown(
        "- **Curva del sistema:** $H(Q)=H_0 + K\\,(Q/3600)^2$, con $Q$ en m³/h y $H$ en m.\n"
        "- **Afinidad:** $Q\\propto n_p$ en 25–50 Hz ⇒ mapeamos linealmente el caudal al rango de velocidades de la bomba.\n"
        "- **Potencia hidráulica:** $P_h = \\dfrac{\\rho g\\,Q_s\\,H(Q)}{\\eta(Q)}$, con $Q_s=Q/3600$ (m³/s).\n"
        "- **Par de bomba:** $T_{pump} = P_h/\\omega_p$, $\\omega_p=2\\pi n_p/60$.\n"
        "- **Reflejo al motor:** $T_{load,m}=T_{pump}/r$.\n"
        "- **Dinámica (eje motor):** $\\dot\\omega_m=\\dfrac{T_{disp}-T_{load,m}}{J_{eq}}$ y el tiempo se integra como "
        "$\\Delta t = J_{eq}\\,\\Delta\\omega_m / (T_{disp}-T_{load,m})$.\n"
        "- **Eficiencia de bomba:** si existen coeficientes $(a,b,c)$ se usa $\\eta(Q)=a+b\\beta+c\\beta^2$, "
        "con $\\beta=Q/Q_{ref}$ y recorte al rango especificado; si no, se asume un valor constante prudente.\n"
    )

# Fin
