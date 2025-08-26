# app.py
# ─────────────────────────────────────────────────────────────────────────────
# Dashboard: Tiempo de reacción de bombas con VDF
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import io
import math
import re
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

# Paleta
BLUE = "#1f77b4"   # Dado (dataset) → azul
GREEN = "#2ca02c"  # Calculado      → verde
GRAY = "#6c757d"

# =============================================================================
# Utilidades de ruta
# =============================================================================

def dataset_path() -> Path:
    return Path(__file__).with_name("dataset.csv")

def images_path(name: str) -> Path:
    return Path(__file__).with_name("images") / name

# =============================================================================
# Formateo y colores
# =============================================================================

def fmt_num(x, unit: str = "", ndigits: int = 2) -> str:
    """Formatea números con coma decimal; devuelve '—' si NaN/None."""
    if isinstance(x, str):
        return f"{x} {unit}".strip()
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    s = f"{x:,.{ndigits}f}"
    s = s.replace(",", "_").replace(".", ",").replace("_", ".")
    return f"{s} {unit}".strip()

def color_value(text: str, color: str, bold: bool = True) -> str:
    w = "600" if bold else "400"
    return f'<span style="color:{color}; font-weight:{w}">{text}</span>'

def val_blue(x, unit: str = "", ndigits: int = 2) -> str:
    return color_value(fmt_num(x, unit, ndigits), BLUE)

def val_green(x, unit: str = "", ndigits: int = 2) -> str:
    return color_value(fmt_num(x, unit, ndigits), GREEN)

# =============================================================================
# Lectura robusta y mapeo del dataset
# =============================================================================

# Mapeo 1:1 contra dataset.csv (sep=';' decimal=',')
ATTR = {
    # Identificación / transmisión
    "TAG":                 {"col": "TAG",                  "unit": "",        "type": "str"},
    "r":                   {"col": "r_trans",              "unit": "",        "type": "num"},  # r = n_motor / n_bomba
    "series":              {"col": "series",               "unit": "",        "type": "str"},
    "grooves":             {"col": "grooves",              "unit": "",        "type": "str"},
    "centerdistance_mm":   {"col": "centerdistance_mm",    "unit": "mm",      "type": "num"},

    # Geometría poleas / ejes (TB Woods)
    "driver_od_in":        {"col": "driver_od_in",         "unit": "in",      "type": "num"},
    "driver_bushing":      {"col": "driver_bushing",       "unit": "",        "type": "str"},
    "driver_shaft_mm":     {"col": "driver_shaft_mm",      "unit": "mm",      "type": "num"},
    "driven_od_in":        {"col": "driven_od_in",         "unit": "in",      "type": "num"},
    "driven_weight_lb":    {"col": "driven_weight_lb",     "unit": "lb",      "type": "num"},
    "driven_bushing":      {"col": "driven_bushing",       "unit": "",        "type": "str"},
    "driven_shaft_mm":     {"col": "driven_shaft_mm",      "unit": "mm",      "type": "num"},

    # Bomba / motor
    "pump_model":          {"col": "pumpmodel",            "unit": "",        "type": "str"},
    "motorpower_kw":       {"col": "motorpower_kw",        "unit": "kW",      "type": "num"},
    "poles":               {"col": "poles",                "unit": "",        "type": "num"},
    "t_nom_nm":            {"col": "t_nom_nm",             "unit": "N·m",     "type": "num"},
    "J_m":                 {"col": "motor_j_kgm2",         "unit": "kg·m²",   "type": "num"},

    # Impulsor (Metso)
    "impeller_d_mm":       {"col": "impeller_d_mm",        "unit": "mm",      "type": "num"},
    "impeller_mass_kg":    {"col": "impeller_mass_kg",     "unit": "kg",      "type": "num"},
    "J_imp":               {"col": "impeller_j_kgm2",      "unit": "kg·m²",   "type": "num"},

    # Velocidades (25–50 Hz)
    "n_m_min":             {"col": "motor_n_min_rpm",      "unit": "rpm",     "type": "num"},
    "n_m_max":             {"col": "motor_n_max_rpm",      "unit": "rpm",     "type": "num"},
    "n_p_min":             {"col": "pump_n_min_rpm",       "unit": "rpm",     "type": "num"},
    "n_p_max":             {"col": "pump_n_max_rpm",       "unit": "rpm",     "type": "num"},
    "n_ref_rpm":           {"col": "n_ref_rpm",            "unit": "rpm",     "type": "num"},

    # Poleas y BUSHINGS (TB Woods)
    "driverpulley_weight_kg": {"col": "driverpulley_weight_kg", "unit": "kg",   "type": "num"},
    "J_driver":            {"col": "driverpulley_j_kgm2",  "unit": "kg·m²",   "type": "num"},
    "J_bushing_driver":    {"col": "driverbushing_j_kgm2", "unit": "kg·m²",   "type": "num"},  # Bushing (antes Sleeve)
    "J_driven":            {"col": "drivenpulley_j_kgm2",  "unit": "kg·m²",   "type": "num"},
    "J_bushing_driven":    {"col": "drivenbushing_j_kgm2", "unit": "kg·m²",   "type": "num"},

    # Curva del sistema H(Q)=H0+K(Q/3600)^2
    "H0_m":                {"col": "H0_m",                 "unit": "m",       "type": "num"},
    "K_m_s2":              {"col": "K_m_s2",               "unit": "",        "type": "num"},
    "R2_H":                {"col": "R2_H",                 "unit": "",        "type": "num"},

    # Eficiencia η(Q)=η_a+η_bβ+η_cβ² (β=Q/Q_ref)
    "eta_a":               {"col": "eta_a",                "unit": "",        "type": "num"},
    "eta_b":               {"col": "eta_b",                "unit": "",        "type": "num"},
    "eta_c":               {"col": "eta_c",                "unit": "",        "type": "num"},
    "R2_eta":              {"col": "R2_eta",               "unit": "",        "type": "num"},
    "Q_min_m3h":           {"col": "Q_min_m3h",            "unit": "m³/h",    "type": "num"},
    "Q_max_m3h":           {"col": "Q_max_m3h",            "unit": "m³/h",    "type": "num"},
    "Q_ref_m3h":           {"col": "Q_ref_m3h",            "unit": "m³/h",    "type": "num"},
    "eta_beta":            {"col": "eta_beta",             "unit": "",        "type": "num"},
    "eta_min_clip":        {"col": "eta_min_clip",         "unit": "",        "type": "num"},
    "eta_max_clip":        {"col": "eta_max_clip",         "unit": "",        "type": "num"},

    # Densidades
    "rho_kgm3":            {"col": "rho_kgm3",             "unit": "kg/m³",   "type": "num"},
    "SlurryDensity":       {"col": "SlurryDensity_kgm3",   "unit": "kg/m³",   "type": "num"},
}

def get_num(x, default=np.nan) -> float:
    """Convierte robustamente números (decimal ','), elimina símbolos/unidades."""
    if x is None:
        return default
    if isinstance(x, (int, float, np.number)):
        try:
            return float(x)
        except Exception:
            return default
    s = str(x).strip()
    if s == "":
        return default
    s = s.replace(" ", "").replace("\u00a0", "")
    s = s.replace(",", ".")
    s = re.sub(r"[^0-9eE\+\-\.]", "", s)
    try:
        return float(s)
    except Exception:
        return default

def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Lee dataset.csv preservando strings; conversión numérica se aplica al leer valores."""
    p = Path(path) if path else dataset_path()
    df = pd.read_csv(p, sep=";", dtype=str, encoding="utf-8", keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    return df

def get_val(row: pd.Series, key: str, default=np.nan):
    meta = ATTR.get(key)
    if not meta:
        return default
    col = meta["col"]
    if col not in row:
        return default
    raw = row[col]
    if meta["type"] == "num":
        return get_num(raw, default)
    return str(raw).strip()

def get_str(row: pd.Series, key: str, default="—") -> str:
    v = get_val(row, key, np.nan)
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return default
    return str(v)

# =============================================================================
# Helpers de negocio (rpm, slider de caudal, hidráulica/η)
# =============================================================================

def rpm_bounds(row: pd.Series):
    """
    Rangos de rpm motor/bomba (min, max) y relación r.
    Si faltan, se deducen con r.
    """
    n_m_min = get_val(row, "n_m_min", np.nan)
    n_m_max = get_val(row, "n_m_max", np.nan)
    n_p_min = get_val(row, "n_p_min", np.nan)
    n_p_max = get_val(row, "n_p_max", np.nan)
    r = get_val(row, "r", np.nan)

    if np.isnan(r) or r <= 0:
        r = np.nan

    if np.isnan(n_m_min) and not np.isnan(r) and not np.isnan(n_p_min):
        n_m_min = r * n_p_min
    if np.isnan(n_m_max) and not np.isnan(r) and not np.isnan(n_p_max):
        n_m_max = r * n_p_max
    if np.isnan(n_p_min) and not np.isnan(r) and not np.isnan(n_m_min):
        n_p_min = n_m_min / r
    if np.isnan(n_p_max) and not np.isnan(r) and not np.isnan(n_m_max):
        n_p_max = n_m_max / r

    # Fallback razonable
    if np.isnan(n_p_min) or np.isnan(n_p_max):
        n_ref = get_val(row, "n_ref_rpm", 1000.0)
        n_p_min = np.nan_to_num(n_p_min, nan=0.5 * n_ref)
        n_p_max = np.nan_to_num(n_p_max, nan=1.0 * n_ref)
    if np.isnan(n_m_min) or np.isnan(n_m_max):
        if not np.isnan(r) and r > 0:
            n_m_min = n_p_min * r
            n_m_max = n_p_max * r
        else:
            n_m_min = 2.0 * n_p_min
            n_m_max = 2.0 * n_p_max

    return float(n_m_min), float(n_m_max), float(n_p_min), float(n_p_max), float(np.nan_to_num(r, nan=1.0))

def flow_slider_bounds(row: pd.Series):
    """
    Límite del slider de caudal por TAG:
    - Mínimo: 0.70 * Q_min_m3h
    - Máximo: 1.30 * Q_max_m3h
    - Por defecto: (Q_min_m3h, Q_max_m3h)
    """
    qmin = get_val(row, "Q_min_m3h", np.nan)
    qmax = get_val(row, "Q_max_m3h", np.nan)
    qref = get_val(row, "Q_ref_m3h", np.nan)

    if np.isnan(qmin) and not np.isnan(qref):
        qmin = 0.7 * qref
    if np.isnan(qmax) and not np.isnan(qref):
        qmax = 1.3 * qref
    if np.isnan(qmin) or np.isnan(qmax) or qmin <= 0 or qmax <= 0:
        qmin, qmax = 100.0, 300.0

    slider_min = max(0.0, 0.70 * qmin)
    slider_max = 1.30 * qmax
    default_low, default_high = float(qmin), float(qmax)
    return float(slider_min), float(slider_max), (default_low, default_high)

def system_head_H(row: pd.Series, Q_m3h: np.ndarray) -> np.ndarray:
    """H(Q) = H0 + K*(Q/3600)^2, con K=K_m_s2 del dataset."""
    H0 = get_val(row, "H0_m", 0.0)
    K  = get_val(row, "K_m_s2", 0.0)
    q_s = np.asarray(Q_m3h, dtype=float) / 3600.0
    return H0 + K * (q_s ** 2)

def eta_from_Q(row: pd.Series, Q_m3h: np.ndarray) -> np.ndarray:
    """
    η(Q) = η_a + η_b*β + η_c*β² con β=Q/Q_ref (si hay coeficientes).
    Si no, usa 'eta_beta' o 0.72 por defecto. Se recorta a [eta_min_clip, eta_max_clip] o [0.40, 0.88].
    """
    Q_ref = get_val(row, "Q_ref_m3h", np.nan)
    a = get_val(row, "eta_a", np.nan)
    b = get_val(row, "eta_b", np.nan)
    c = get_val(row, "eta_c", np.nan)
    eta_beta = get_val(row, "eta_beta", np.nan)

    if not any(np.isnan(v) for v in (a, b, c, Q_ref)) and Q_ref > 0:
        beta = np.asarray(Q_m3h, dtype=float) / Q_ref
        eta = a + b * beta + c * (beta ** 2)
    elif not np.isnan(eta_beta):
        eta = np.full_like(Q_m3h, float(eta_beta), dtype=float)
    else:
        eta = np.full_like(Q_m3h, 0.72, dtype=float)

    eta_min = get_val(row, "eta_min_clip", 0.40)
    eta_max = get_val(row, "eta_max_clip", 0.88)
    return np.clip(eta, eta_min, eta_max)

def hydraulic_torque_on_motor(row: pd.Series,
                              n_p_rpm: np.ndarray,
                              Q_low_high: tuple[float, float],
                              rho_kgm3: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Devuelve (T_load_m, P_h, eta, Q) a lo largo de un perfil de n_p_rpm.
    - Q por afinidad lineal entre n_p_min y n_p_max hacia (Q_low, Q_high) del slider
    - H(Q) del sistema
    - P_h = rho*g*Q_s*H / η
    - T_pump = P_h / ω_p
    - T_load_m = T_pump / r
    """
    g = 9.81
    rho_data = get_val(row, "SlurryDensity", np.nan)
    rho = rho_kgm3 if (rho_kgm3 and rho_kgm3 > 0) else (rho_data if not np.isnan(rho_data) else get_val(row, "rho_kgm3", 1000.0))
    r = get_val(row, "r", 1.0)
    _, _, n_p_min, n_p_max, _ = rpm_bounds(row)

    q_low, q_high = Q_low_high
    if n_p_max <= n_p_min + 1e-9:
        Q = np.full_like(n_p_rpm, q_low)
    else:
        Q = np.interp(n_p_rpm, [n_p_min, n_p_max], [q_low, q_high])

    H = system_head_H(row, Q)
    eta = eta_from_Q(row, Q)
    Qs = Q / 3600.0
    P_h = rho * g * Qs * H / np.maximum(eta, 1e-6)  # W
    omega_p = 2.0 * np.pi * np.asarray(n_p_rpm, dtype=float) / 60.0
    T_pump = np.where(omega_p > 1e-9, P_h / omega_p, 0.0)
    T_load_m = T_pump / max(r, 1e-9)
    return T_load_m, P_h, eta, Q

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

df = load_data()
tags = df["TAG"].astype(str).tolist()
tag_sel = st.sidebar.selectbox("Selecciona TAG", tags, index=0)
row = df[df["TAG"].astype(str) == str(tag_sel)].iloc[0]

# Rango de rpm y relación (para usar en varias secciones)
n_m_min, n_m_max, n_p_min, n_p_max, r = rpm_bounds(row)

# Parámetros base
pump_model = get_str(row, "pump_model")
P_motor_kW = get_val(row, "motorpower_kw")
T_nom_nm   = get_val(row, "t_nom_nm")
D_imp_mm   = get_val(row, "impeller_d_mm")

# Densidad
rho_slurry = get_val(row, "SlurryDensity", np.nan)
rho_base   = get_val(row, "rho_kgm3", 1000.0)
rho_use    = rho_slurry if not np.isnan(rho_slurry) else rho_base

# =============================================================================
# 1) Parámetros
# =============================================================================

st.markdown("## 1) Parámetros")

c1, c2, c3 = st.columns([1.2, 1.2, 1.2])

with c1:
    st.markdown("**Identificación**")
    st.markdown(f"- Modelo de bomba: {val_blue(pump_model, '', 0)}", unsafe_allow_html=True)
    st.markdown(f"- TAG: {val_blue(tag_sel, '', 0)}", unsafe_allow_html=True)
    st.markdown(f"- Diámetro impulsor: {val_blue(D_imp_mm, 'mm')}", unsafe_allow_html=True)

with c2:
    st.markdown("**Motor & transmisión**")
    st.markdown(f"- Potencia motor instalada: {val_blue(P_motor_kW, 'kW')}", unsafe_allow_html=True)
    st.markdown(f"- Par nominal del motor: {val_blue(T_nom_nm, 'N·m')}", unsafe_allow_html=True)
    st.markdown(r"- Relación de transmisión $r = \dfrac{n_{\mathrm{motor}}}{n_{\mathrm{bomba}}}$: " + f"{val_blue(r, '')}", unsafe_allow_html=True)
    st.markdown("- Velocidad motor min–max: " + f"{val_blue(n_m_min, 'rpm', 0)} – {val_blue(n_m_max, 'rpm', 0)}", unsafe_allow_html=True)

with c3:
    st.markdown("**Bomba (25–50 Hz por afinidad)**")
    st.markdown("- Velocidad bomba min–max: " + f"{val_blue(n_p_min, 'rpm', 0)} – {val_blue(n_p_max, 'rpm', 0)}", unsafe_allow_html=True)
    st.markdown(f"- Densidad de pulpa ρ: {val_blue(rho_use, 'kg/m³', 0)}", unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# 2) Cálculo de inercia equivalente
# =============================================================================

st.header("2) Cálculo de inercia equivalente")

colL, colR = st.columns([1.1, 1])

with colL:
    st.subheader("Inercias individuales")

    J_m               = get_val(row, "J_m", 0.0)
    J_driver          = get_val(row, "J_driver", 0.0)
    J_bushing_driver  = get_val(row, "J_bushing_driver", np.nan)
    if np.isnan(J_bushing_driver):
        J_bushing_driver = 0.10 * J_driver  # fallback 10% catálogo TB Woods

    J_driven          = get_val(row, "J_driven", 0.0)
    J_bushing_driven  = get_val(row, "J_bushing_driven", np.nan)
    if np.isnan(J_bushing_driven):
        J_bushing_driven = 0.10 * J_driven   # fallback 10%

    J_imp             = get_val(row, "J_imp", 0.0)

    st.markdown(f"- Motor (J_m): {val_blue(J_m, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Polea motriz (J_driver): {val_blue(J_driver, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Bushing motriz (≈10% J_driver): {val_blue(J_bushing_driver, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Polea conducida (J_driven): {val_blue(J_driven, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Bushing conducido (≈10% J_driven): {val_blue(J_bushing_driven, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Impulsor/rotor de bomba (J_imp): {val_blue(J_imp, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(r"- Relación $r = \dfrac{n_m}{n_p}$: " + f"{val_blue(r, '')}", unsafe_allow_html=True)

    # J_eq en el eje del motor (inercias del lado bomba vistas /r²)
    J_eq = (J_m + J_driver + J_bushing_driver) + (J_driven + J_bushing_driven + J_imp) / (r**2)
    st.markdown(f"**Inercia equivalente (J_eq):** {val_green(J_eq, 'kg·m²')}", unsafe_allow_html=True)

with colR:
    st.subheader("Fórmulas")
    st.latex(r"J_{\mathrm{eq}} = J_m + J_{\mathrm{driver}} + J_{\mathrm{bushing,driver}} + \frac{J_{\mathrm{driven}} + J_{\mathrm{bushing,driven}} + J_{\mathrm{imp}}}{r^2}")
    st.markdown("**Justificación del $r^2$:**")
    st.latex(r"\omega_p = \frac{\omega_m}{r}")
    st.latex(r"E_k = \tfrac{1}{2}J\,\omega^2 \;\Rightarrow\; J_{\text{lado bomba}} \text{ se ve como } \frac{J}{r^2} \text{ en el eje motor}")
    with st.expander("Origen de datos por componente", expanded=True):
        st.markdown(
            "- **Poleas** (J_driver, J_driven): catálogo **TB Woods**.\n"
            "- **Bushings**: si no hay dato, se aproxima **10%** de la inercia de su polea.\n"
            "- **Impulsor** (J_imp): manuales **Metso**.\n"
        )

st.markdown("---")

# =============================================================================
# 3) Tiempo inercial (par disponible vs rampa VDF)
# =============================================================================

st.markdown("## 3) Tiempo inercial (par disponible vs rampa VDF)")

rampa_vdf = st.slider("Rampa VDF en el motor [rpm/s]", min_value=10, max_value=600, value=100, step=5)

# Aceleración por par (suponiendo T_disp = T_nom y pérdidas mecánicas despreciables)
if not (np.isnan(J_eq) or np.isnan(T_nom_nm) or J_eq <= 0):
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

def pill(text: str, bg: str = "#e8f5e9", color: str = "#1b5e20"):
    st.markdown(
        f"""
        <div style="border-left: 5px solid {color}; background:{bg}; padding:0.8rem 1rem; border-radius:0.5rem; margin-top:0.5rem">
            <span style="color:{color}; font-weight:700">{text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

lim3 = "Tiempo limitante (sección 3): "
if not np.isnan(t_par) and t_par > t_rampa:
    pill(lim3 + f"por par = {fmt_num(t_par, 's')}")
else:
    pill(lim3 + f"por rampa VDF = {fmt_num(t_rampa, 's')}")

with st.expander("Detalles y fórmulas — Sección 3", expanded=False):
    st.markdown("**Hipótesis:** par del motor constante y pérdidas mecánicas despreciables.")
    st.latex(r"T_{\mathrm{disp}} = T_{\mathrm{nom}}")
    st.latex(r"J_{\mathrm{eq}}\,\dot\omega_m = T_{\mathrm{disp}} \;\Rightarrow\; \dot\omega_m = \tfrac{T_{\mathrm{disp}}}{J_{\mathrm{eq}}}")
    st.latex(r"n_m = \tfrac{60}{2\pi}\,\omega_m \;\Rightarrow\; \dot n_m = \tfrac{60}{2\pi}\,\tfrac{T_{\mathrm{disp}}}{J_{\mathrm{eq}}}")
    st.latex(r"t_{\mathrm{par}} = \dfrac{\Delta n_m}{\dot n_m}, \qquad t_{\mathrm{rampa}} = \dfrac{\Delta n_m}{\text{rampa}}")
    st.latex(r"\text{Criterio: } t = \max\{t_{\mathrm{par}},\,t_{\mathrm{rampa}}\}")

st.markdown("---")

# =============================================================================
# 4) Integración con carga hidráulica
# =============================================================================

st.markdown("## 4) Integración con carga hidráulica")

# Fórmulas intro (H, η y la integración)
st.latex(r"H(Q) = H_0 + K\,\big(\tfrac{Q}{3600}\big)^2 \qquad [\,Q:\ \mathrm{m^3/h},\ H:\ \mathrm{m}\,]")
st.latex(r"\eta(Q) = \eta_a + \eta_b\,\beta + \eta_c\,\beta^2,\quad \beta = \tfrac{Q}{Q_{\mathrm{ref}}},\quad \eta\in[\eta_{\min},\eta_{\max}]")
st.latex(r"P_h = \rho\,g\,Q_s\,H(Q)\,/\,\eta(Q), \quad Q_s=\tfrac{Q}{3600}")
st.latex(r"T_{\mathrm{pump}} = \tfrac{P_h}{\omega_p},\ \ \omega_p=\tfrac{2\pi n_p}{60};\qquad T_{\mathrm{load,m}}=\tfrac{T_{\mathrm{pump}}}{r}")
st.latex(r"T_{\mathrm{net}} = T_{\mathrm{disp}} - T_{\mathrm{load,m}};\quad \Delta t = \tfrac{J_{\mathrm{eq}}\ \Delta\omega_m}{T_{\mathrm{net}}}")
st.latex(r"\text{Tiempo total: } \ \int_{n_{m,\min}}^{n_{m,\max}} \frac{J_{\mathrm{eq}}(2\pi/60)}{T_{\mathrm{net}}(n_m)}\, dn_m")

# Parámetros hidráulicos de cabecera
H0_m     = get_val(row, "H0_m")
K_m_s2   = get_val(row, "K_m_s2")
eta_min  = get_val(row, "eta_min_clip", 0.40)
eta_max  = get_val(row, "eta_max_clip", 0.88)
Q_ref    = get_val(row, "Q_ref_m3h")

c4a, c4b, c4c, c4d = st.columns(4)
with c4a:
    st.markdown(f"- H₀: {val_blue(H0_m, 'm')}", unsafe_allow_html=True)
with c4b:
    st.markdown(f"- K (K_m_s2): {val_blue(K_m_s2, '')}", unsafe_allow_html=True)
with c4c:
    st.markdown(f"- ρ: {val_blue(rho_use, 'kg/m³', 0)}", unsafe_allow_html=True)
with c4d:
    st.markdown(f"- η (clip): {val_blue(eta_min*100, '%', 0)} – {val_blue(eta_max*100, '%', 0)}", unsafe_allow_html=True)

# Slider de caudal que se adapta a cada TAG (−30% de Qmin a +30% de Qmax)
slider_min, slider_max, default_range = flow_slider_bounds(row)
Q_min, Q_max = st.slider(
    "Rango de caudal considerado [m³/h]",
    min_value=float(slider_min),
    max_value=float(slider_max),
    value=(float(default_range[0]), float(default_range[1])),
    step=1.0,
    key=f"q_slider_{tag_sel}",  # clave dependiente del TAG para que cambie al cambiar TAG
)

# Mallas para integración
N = 600
n_m_grid = np.linspace(n_m_min, n_m_max, N)
n_p_grid = np.linspace(n_p_min, n_p_max, N)

# Par resistente reflejado + potencia + eficiencia + Q(n_p)
T_load_m, P_h_W, eta_grid, Q_grid = hydraulic_torque_on_motor(row, n_p_grid, (Q_min, Q_max), rho_kgm3=rho_use)
T_disp_m = np.full_like(T_load_m, T_nom_nm)

# Gráfico 1: Par en el eje motor vs Velocidad bomba (sin línea roja 0)
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

# Gráfico 2: Potencia hidráulica vs Velocidad bomba
fig_p = go.Figure()
fig_p.add_trace(go.Scatter(x=n_p_grid, y=P_h_W/1000.0, mode="lines", name="P hidráulica [kW]"))
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
    # Para alinear T_net con d_omega, evaluamos T_net en puntos intermedios
    # Mapear n_m -> n_p linealmente (misma fracción de progreso)
    frac = (n_m_grid - n_m_min) / max((n_m_max - n_m_min), 1e-9)
    n_p_interp = n_p_min + frac * (n_p_max - n_p_min)
    # Recalcular T_load en esos puntos intermedios (opcionalmente, usar promedio)
    T_load_m_i, _, _, _ = hydraulic_torque_on_motor(row, n_p_interp, (Q_min, Q_max), rho_kgm3=rho_use)
    T_disp_i = np.full_like(T_load_m_i, T_nom_nm)
    T_net_i = T_disp_i - T_load_m_i
    T_net_clip = np.maximum(T_net_i[:-1], 1e-9)
    dt = (J_eq * d_omega) / T_net_clip
    t_hyd = float(np.sum(dt))
else:
    t_hyd = float("nan")

c4_1, c4_2 = st.columns(2)
with c4_1:
    st.markdown(f"- Tiempo por carga **hidráulica** (integración): {val_green(t_hyd, 's')}", unsafe_allow_html=True)
with c4_2:
    t_lim_4 = max(t_hyd, t_rampa) if not np.isnan(t_hyd) else t_rampa
    which = "hidráulica" if (not np.isnan(t_hyd) and t_hyd > t_rampa) else "rampa VDF"
    # Mostrar justo debajo del tiempo hidráulico
    pill(f"Tiempo limitante (sección 4): {which} = {fmt_num(t_lim_4, 's')}")

with st.expander("Detalles y fórmulas — Sección 4", expanded=False):
    st.markdown("**Curva del sistema:**")
    st.latex(r"H(Q)=H_0+K\,(Q/3600)^2 \quad\text{con}\quad K=K_{\mathrm{m\_s2}}")
    st.markdown("**Afinidad (25–50 Hz):**")
    st.latex(r"Q \propto n_p,\ \ \text{interpolado entre } n_{p,\min} \text{ y } n_{p,\max}")
    st.markdown("**Eficiencia:**")
    st.latex(r"\eta(Q)=\eta_a+\eta_b\,\beta+\eta_c\,\beta^2,\ \ \beta=Q/Q_{\mathrm{ref}};\ \ \text{si no hay coeficientes, usar }\eta\ \text{constante del dataset}")
    st.latex(r"\eta \in [0.40,\,0.88]")
    st.markdown("**Potencia, par e integración temporal:**")
    st.latex(r"P_h = \rho g Q_s H(Q)/\eta(Q),\quad T_{\mathrm{pump}}=P_h/\omega_p,\quad T_{\mathrm{load,m}}=T_{\mathrm{pump}}/r")
    st.latex(r"T_{\mathrm{net}}=T_{\mathrm{disp}}-T_{\mathrm{load,m}},\quad \Delta t = J_{\mathrm{eq}}\Delta\omega_m/T_{\mathrm{net}}")
    st.latex(r"\text{Tiempo total } t = \int_{n_{m,\min}}^{n_{m,\max}} \frac{J_{\mathrm{eq}}(2\pi/60)}{T_{\mathrm{net}}(n_m)}\, dn_m")

st.markdown("---")

# =============================================================================
# 5) Exportar CSV con resultados del TAG
# =============================================================================

st.markdown("## 5) Exportar resultados del TAG a CSV")

# Construir dataframe de resultados (serie a lo largo de la rampa 25→50 Hz)
# Usaremos el grid ya calculado:
# - n_p_grid, n_m_grid, Q_grid, H(Q), η, P_h (kW), T_load_m, T_disp, T_net, omega_m, dt (diferencial)
H_grid = system_head_H(row, Q_grid)
omega_m_grid = 2.0 * math.pi * n_m_grid / 60.0
T_net = T_disp_m - T_load_m

# dt con el método de integración usado (usar mismos T_net_i que para t_hyd)
d_omega = np.diff(omega_m_grid)
T_net_clip = np.maximum(T_net[:-1], 1e-9)
dt_series = (J_eq * d_omega) / T_net_clip
dt_series = np.append(dt_series, dt_series[-1] if len(dt_series) else 0.0)

df_out = pd.DataFrame({
    "TAG":        [tag_sel]*len(n_p_grid),
    "n_p_rpm":    n_p_grid,
    "n_m_rpm":    n_m_grid,
    "Q_m3h":      Q_grid,
    "H_m":        H_grid,
    "eta":        eta_grid,
    "P_h_kW":     P_h_W/1000.0,
    "T_load_m_Nm":T_load_m,
    "T_disp_Nm":  T_disp_m,
    "T_net_Nm":   T_net,
    "omega_m_rad_s": omega_m_grid,
    "dt_s":       dt_series,
})

# Resumen calculado
summary = {
    "J_eq_kgm2": J_eq,
    "r_nm_np":   r,
    "n_m_min_rpm": n_m_min,
    "n_m_max_rpm": n_m_max,
    "n_p_min_rpm": n_p_min,
    "n_p_max_rpm": n_p_max,
    "T_nom_Nm":  T_nom_nm,
    "rampa_vdf_rpmps": rampa_vdf,
    "t_par_s":   t_par,
    "t_rampa_s": t_rampa,
    "t_hid_s":   t_hyd,
    "rho_kgm3":  rho_use,
    "H0_m":      H0_m,
    "K_m_s2":    K_m_s2,
    "Q_slider_min_m3h": Q_min,
    "Q_slider_max_m3h": Q_max,
}

# Exportar a CSV (con separador coma y punto decimal)
csv_buf = io.StringIO()
# Escribimos primero una cabecera de resumen (clave,valor)
pd.Series(summary).to_csv(csv_buf, header=False)
csv_buf.write("\n")  # línea en blanco
df_out.to_csv(csv_buf, index=False)
csv_bytes = csv_buf.getvalue().encode("utf-8")

st.download_button(
    "⬇️ Descargar CSV del TAG (resumen + serie 25→50 Hz)",
    data=csv_bytes,
    file_name=f"reporte_{tag_sel}.csv",
    mime="text/csv",
)

# Fin
