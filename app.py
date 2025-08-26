# -*- coding: utf-8 -*-
"""
App: Tiempo de reacción de bombas (VDF) - versión básica (sin hidráulica)
Suposiciones clave:
- El archivo 'dataset.csv' está en el mismo directorio del app (repo), con sep=';' y decimal=','.
- Las columnas del dataset siguen el mapeo definido en FIELDS (ver abajo). Se hace conversión robusta de tipos.
- Rango de análisis de velocidad: 25 → 50 Hz (ajustable con n_nominal, por defecto n_50 = n_nominal).
"""

import re
import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, Any

st.set_page_config(
    page_title="Memoria de Cálculo – Tiempo de reacción (VDF)",
    page_icon="⏱️",
    layout="wide"
)

# ──────────────────────────────────────────────────────────────────────────────
#  Diccionario de campos: descripción, unidad y tipo de dato esperado
#  type ∈ {"string", "category", "int", "float"}
# ──────────────────────────────────────────────────────────────────────────────

FIELDS: Dict[str, Dict[str, Any]] = {
    # Identificación
    "tag":               {"desc": "TAG de la bomba", "unit": "",        "type": "category"},
    "pumpmodel":         {"desc": "Modelo de bomba", "unit": "",        "type": "string"},
    "series":            {"desc": "Serie/familia",    "unit": "",       "type": "category"},

    # Transmisión
    "driver_bushing":    {"desc": "Buje polea motriz",    "unit": "",   "type": "category"},
    "driven_bushing":    {"desc": "Buje polea conducida", "unit": "",   "type": "category"},
    "driver_od_in":      {"desc": "Ø polea motriz",       "unit": "in", "type": "float"},
    "driven_od_in":      {"desc": "Ø polea conducida",    "unit": "in", "type": "float"},
    "driver_shaft_mm":   {"desc": "Eje motriz",           "unit": "mm", "type": "float"},
    "driven_shaft_mm":   {"desc": "Eje conducido",        "unit": "mm", "type": "float"},
    "grooves":           {"desc": "Nº ranuras correa",    "unit": "",   "type": "int"},
    "driven_weight_lb":  {"desc": "Peso polea conducida", "unit": "lb", "type": "float"},
    "r_trans":           {"desc": "Relación transmisión (r = n_motor/n_bomba)", "unit": "", "type": "float"},

    # Motor
    "motorpower_kw":     {"desc": "Potencia motor instalada", "unit": "kW",  "type": "float"},
    "poles":             {"desc": "Nº de polos motor",        "unit": "",    "type": "int"},
    "t_nom_nm":          {"desc": "Par nominal motor",         "unit": "N·m", "type": "float"},
    "n_min_rpm":         {"desc": "Velocidad mínima motor (placa)", "unit": "rpm", "type": "float"},
    "n_nom_rpm":         {"desc": "Velocidad nominal motor",       "unit": "rpm", "type": "float"},
    "n_max_rpm":         {"desc": "Velocidad máxima motor (placa)", "unit": "rpm", "type": "float"},
    "motor_j_kgm2":      {"desc": "Inercia motor", "unit": "kg·m²", "type": "float"},

    # Inercias adicionales
    "jdriver_kgm2":      {"desc": "Inercia polea motriz",   "unit": "kg·m²", "type": "float"},
    "jdriven_kgm2":      {"desc": "Inercia polea conducida","unit": "kg·m²", "type": "float"},
    "jimp_kgm2":         {"desc": "Inercia impulsor/rotor bomba", "unit": "kg·m²", "type": "float"},
    "m_motor_kg":        {"desc": "Masa motor",     "unit": "kg", "type": "float"},
    "m_driver_kg":       {"desc": "Masa polea motriz", "unit": "kg", "type": "float"},
    "m_coupling_kg":     {"desc": "Masa acople",    "unit": "kg", "type": "float"},
    "n_wheel":           {"desc": "Nº de etapas/rodetes", "unit": "", "type": "int"},

    # Pérdidas mecánicas (no usadas aquí, reservado)
    "mu_box":            {"desc": "Pérdida caja",       "unit": "", "type": "float"},
    "mu_vbelt":          {"desc": "Pérdida correa V",   "unit": "", "type": "float"},
    "mu_htd":            {"desc": "Pérdida correa HTD", "unit": "", "type": "float"},

    # Curva hidráulica y eficiencia (reservado para etapa 2 con hidráulica)
    "h0_m":              {"desc": "Altura en Q=0",       "unit": "m",        "type": "float"},
    "k_m_s2":            {"desc": "Coef. cuadrático H(Q) con Q en m³/h", "unit": "m/(m³/h)²", "type": "float"},
    "r2_h":              {"desc": "R² ajuste H(Q)",      "unit": "",         "type": "float"},
    "q_min_m3h":         {"desc": "Q mínima",            "unit": "m³/h",     "type": "float"},
    "q_max_m3h":         {"desc": "Q máxima",            "unit": "m³/h",     "type": "float"},
    "q_ref_m3h":         {"desc": "Q referencia",        "unit": "m³/h",     "type": "float"},
    "n_ref_rpm":         {"desc": "n referencia bomba",  "unit": "rpm",      "type": "float"},
    "rho_kgm3":          {"desc": "Densidad fluido",     "unit": "kg/m³",    "type": "float"},
    "eta_a":             {"desc": "η(Q) coef a",         "unit": "",         "type": "float"},
    "eta_b":             {"desc": "η(Q) coef b",         "unit": "",         "type": "float"},
    "eta_c":             {"desc": "η(Q) coef c",         "unit": "",         "type": "float"},
    "r2_eta":            {"desc": "R² ajuste η(Q)",      "unit": "",         "type": "float"},
    "eta_beta":          {"desc": "η(Q) exponente extra","unit": "",         "type": "float"},
    "eta_min_clip":      {"desc": "η mínima (clip)",     "unit": "",         "type": "float"},
    "eta_max_clip":      {"desc": "η máxima (clip)",     "unit": "",         "type": "float"},
}

# ──────────────────────────────────────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────────────────────────────────────

def to_float(x) -> float:
    """Convierte texto con coma/puntos/unidades a float. NaN si vacío o inválido."""
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.number)):
        try:
            return float(x)
        except Exception:
            return np.nan
    s = str(x).strip().replace("\u00a0", " ")  # NBSP → espacio
    if s == "" or s.lower() in {"nan", "none", "null", "-"}:
        return np.nan
    # Quita espacios y unidades, preserva dígitos, signos y e/E
    s = s.replace(" ", "")
    s = s.replace(",", ".")
    s = re.sub(r"[^0-9\.\-eE\+]", "", s)
    try:
        return float(s)
    except Exception:
        return np.nan

def to_int(x) -> int:
    v = to_float(x)
    if np.isnan(v):
        return np.nan
    return int(round(v))

def load_dataset(path: str = "dataset.csv") -> pd.DataFrame:
    # Leemos como texto para luego tipar según FIELDS
    df = pd.read_csv(path, sep=";", decimal=",", dtype=str, encoding="utf-8", engine="python")
    # Normaliza nombres de columnas a minúsculas
    df.columns = [c.strip().lower() for c in df.columns]
    # Asegura existencia de 'tag'
    if "tag" not in df.columns:
        raise ValueError("La columna 'TAG' (o 'tag') es obligatoria en dataset.csv")

    # Tipado según FIELDS
    numeric_cols, int_cols = [], []
    for k, meta in FIELDS.items():
        if meta["type"] in {"float", "int"} and k in df.columns:
            numeric_cols.append(k)
            if meta["type"] == "int":
                int_cols.append(k)

    for c in numeric_cols:
        df[c] = df[c].map(to_float)

    for c in int_cols:
        df[c] = df[c].map(to_int)

    # Convierte categorías
    for k, meta in FIELDS.items():
        if meta["type"] == "category" and k in df.columns:
            df[k] = df[k].astype("category")

    # Limpia ceros imposibles
    for c in ["r_trans", "motor_j_kgm2", "jdriver_kgm2", "jdriven_kgm2", "jimp_kgm2"]:
        if c in df.columns:
            df.loc[df[c] == 0, c] = np.nan

    return df

def get_row(df: pd.DataFrame, tag: str) -> pd.Series:
    r = df.loc[df["tag"] == tag]
    if r.empty:
        raise ValueError(f"TAG '{tag}' no encontrado en dataset.")
    return r.iloc[0]

def ratio_transmision(row: pd.Series) -> float:
    """r = n_motor / n_bomba.
    1) Usa r_trans si existe.
    2) Si no, estima con poleas: r ≈ D_conducida / D_motriz (sin deslizamiento)."""
    r = np.nan
    if "r_trans" in row.index:
        r = to_float(row["r_trans"])
    if not (isinstance(r, float) and r > 0):
        d_drv = to_float(row.get("driver_od_in", np.nan))
        d_dvn = to_float(row.get("driven_od_in", np.nan))
        if d_drv and d_dvn and d_drv > 0:
            r = d_dvn / d_drv  # N_motor/N_bomba = D_conducida/D_motriz
    return r if (isinstance(r, float) and r > 0) else np.nan

def nominal_speed_50hz(row: pd.Series) -> float:
    """Devuelve n_nom_rpm; si falta, lo aproxima con polos a 50 Hz (con ~3% de deslizamiento)."""
    n_nom = to_float(row.get("n_nom_rpm", np.nan))
    if not (isinstance(n_nom, float) and n_nom > 0):
        poles = to_int(row.get("poles", np.nan))
        if poles and poles > 0:
            n_sync = 120.0 * 50.0 / poles
            n_nom = n_sync * 0.97  # aproximación con 3% slip
    return n_nom

def fmt(value, unit=""):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if isinstance(value, (int, np.integer)):
        s = f"{value:d}"
    else:
        s = f"{float(value):,.2f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")  # coma decimal
    return f"{s} {unit}".strip()

# ──────────────────────────────────────────────────────────────────────────────
#  Carga de datos
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_data():
    df = load_dataset("dataset.csv")
    tags = sorted(df["tag"].astype(str).unique().tolist())
    return df, tags

df, TAGS = get_data()

# ──────────────────────────────────────────────────────────────────────────────
#  UI – Barra lateral
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Selección")
    tag = st.selectbox("Selecciona TAG", TAGS, index=0)

    st.markdown("---")
    st.header("🔧 Ajustes de cálculo")
    # Factor de par disponible respecto a par nominal
    torque_factor = st.slider("Par disponible (× T_nom)", min_value=0.50, max_value=2.00, value=1.00, step=0.05)
    # Rampa del VDF en rpm/s
    rampa_vdf = st.slider("Rampa VDF [rpm/s]", min_value=10, max_value=1000, value=200, step=10)

    st.caption("El análisis se realiza entre **25–50 Hz**. El par disponible se asume "
               "constante en ese rango y se multiplica por T_nom del motor.")

row = get_row(df, tag)

# ──────────────────────────────────────────────────────────────────────────────
#  Sección 1: Parámetros
# ──────────────────────────────────────────────────────────────────────────────
st.title("⏱️ Tiempo de reacción (VDF) – Memoria de Cálculo")

st.subheader("1) Parámetros (25–50 Hz)")
col1, col2, col3, col4 = st.columns([1.2, 1, 1, 1])

r = ratio_transmision(row)
n_nom = nominal_speed_50hz(row)
n25 = np.nan
n50 = np.nan
if n_nom and n_nom > 0:
    n25 = 0.5 * n_nom
    n50 = 1.0 * n_nom

with col1:
    st.markdown("**Identificación**")
    st.write(f"- **TAG:** {row.get('tag', '—')}")
    st.write(f"- **Modelo bomba:** {row.get('pumpmodel', '—')}")
    st.write(f"- **Serie:** {row.get('series', '—')}")
with col2:
    st.markdown("**Motor**")
    st.write(f"- **Potencia instalada:** {fmt(row.get('motorpower_kw'), 'kW')}")
    st.write(f"- **Par nominal:** {fmt(row.get('t_nom_nm'), 'N·m')}")
    st.write(f"- **Polos:** {fmt(row.get('poles'))}")
with col3:
    st.markdown("**Transmisión**")
    st.write(f"- **Relación r = nₘ/nₚ:** {fmt(r)}")
    st.write(f"- Ø motriz: {fmt(row.get('driver_od_in'), 'in')}")
    st.write(f"- Ø conducida: {fmt(row.get('driven_od_in'), 'in')}")
with col4:
    st.markdown("**Velocidades (25–50 Hz)**")
    st.write(f"- **Motor:** {fmt(n25, 'rpm')} → {fmt(n50, 'rpm')}")
    if r and r > 0 and n25 and n50:
        st.write(f"- **Bomba:** {fmt(n25 / r, 'rpm')} → {fmt(n50 / r, 'rpm')}")
    else:
        st.write("- **Bomba:** —")

st.caption("Nota: se asume proporcionalidad n ∝ f en 25–50 Hz; si no hay n_nom_rpm se aproxima con polos (50 Hz, ~3 % slip).")

# ──────────────────────────────────────────────────────────────────────────────
#  Sección 2: Cálculo de inercia
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("2) Cálculo de inercia equivalente")

Jm   = to_float(row.get("motor_j_kgm2"))
Jdrv = to_float(row.get("jdriver_kgm2"))
Jdvn = to_float(row.get("jdriven_kgm2"))
Jimp = to_float(row.get("jimp_kgm2"))

if r and r > 0:
    Jeq = (Jm or 0.0) + (Jdrv or 0.0) + ((Jdvn or 0.0) + (Jimp or 0.0)) / (r ** 2)
else:
    Jeq = np.nan

c1, c2 = st.columns([1.1, 1.3])
with c1:
    st.markdown("**Inercias individuales**")
    st.write(f"- Motor \(J_m\): {fmt(Jm, 'kg·m²')}")
    st.write(f"- Polea motriz \(J_{{driver}}\): {fmt(Jdrv, 'kg·m²')}")
    st.write(f"- Polea conducida \(J_{{driven}}\): {fmt(Jdvn, 'kg·m²')}")
    st.write(f"- Impulsor/rotor bomba \(J_{{imp}}\): {fmt(Jimp, 'kg·m²')}")
    st.write(f"- **Relación r** (nₘ/nₚ): {fmt(r)}")
    st.write(f"- **Inercia equivalente \(J_{{eq}}\)**: {fmt(Jeq, 'kg·m²')}")

with c2:
    st.markdown("**Fórmula utilizada**")
    st.latex(r"J_{\mathrm{eq}} = J_m + J_{\mathrm{driver}} + \dfrac{J_{\mathrm{driven}} + J_{\mathrm{imp}}}{r^2}")
    st.caption("Las inercias del lado bomba giran a \( \omega_p = \omega_m / r \). "
               "Igualando energías cinéticas a una \( \omega_m \) común se obtiene la división por \( r^2 \) para términos de la bomba.")

# ──────────────────────────────────────────────────────────────────────────────
#  Sección 3: Tiempo de aceleración (par vs rampa)
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("3) Tiempo de aceleración (par vs rampa)")

# Par disponible (constante 25–50 Hz)
T_nom = to_float(row.get("t_nom_nm"))
T_disp = (T_nom if T_nom and T_nom > 0 else np.nan)
if T_disp and torque_factor:
    T_disp *= torque_factor

# Derivada de rpm por par (dn/dt) en el eje motor
#   α = T / J_eq [rad/s²]
#   dn/dt [rpm/s] = α * 60 / (2π) = T * 60 / (2π J_eq)
if Jeq and Jeq > 0 and T_disp and T_disp > 0:
    dn_dt_torque = T_disp * 60.0 / (2.0 * np.pi * Jeq)  # rpm/s sobre el eje motor
else:
    dn_dt_torque = np.nan

# Δn motor entre 25 y 50 Hz
if n25 and n50:
    delta_n = n50 - n25
else:
    delta_n = np.nan

t_par = delta_n / dn_dt_torque if (isinstance(delta_n, float) and delta_n > 0 and isinstance(dn_dt_torque, float) and dn_dt_torque > 0) else np.nan
t_rampa = delta_n / rampa_vdf if (isinstance(delta_n, float) and delta_n > 0 and rampa_vdf and rampa_vdf > 0) else np.nan

colL, colR = st.columns([1, 1])
with colL:
    st.markdown("**Ecuaciones**")
    st.latex(r"\alpha = \frac{T_{\mathrm{disp}}}{J_{\mathrm{eq}}}\quad [\mathrm{rad/s^2}]")
    st.latex(r"\frac{dn_m}{dt} = \alpha \cdot \frac{60}{2\pi} = \frac{T_{\mathrm{disp}}}{J_{\mathrm{eq}}}\cdot \frac{60}{2\pi}\quad [\mathrm{rpm/s}]")
    st.latex(r"t_{\mathrm{par}} = \dfrac{\Delta n_m}{\,dn_m/dt\,}, \qquad t_{\mathrm{rampa}} = \dfrac{\Delta n_m}{\mathrm{rampa}_{\mathrm{VDF}}}")
    st.caption("Se compara el tiempo por par disponible (constante en 25–50 Hz) con el tiempo por rampa VDF. "
               "El tiempo real estará gobernado por el **más lento** de ambos.")

with colR:
    st.markdown("**Resultados (25→50 Hz sobre eje motor)**")
    st.write(f"- Δn motor: {fmt(delta_n, 'rpm')}")
    st.write(f"- \(dn_m/dt\) por par: {fmt(dn_dt_torque, 'rpm/s')}")
    st.write(f"- **t\_par**: {fmt(t_par, 's')}")
    st.write(f"- Rampa VDF: {fmt(rampa_vdf, 'rpm/s')}")
    st.write(f"- **t\_rampa**: {fmt(t_rampa, 's')}")

# Gobierno
t_real = np.nan
if isinstance(t_par, float) and t_par > 0 and isinstance(t_rampa, float) and t_rampa > 0:
    t_real = max(t_par, t_rampa)
elif isinstance(t_par, float) and t_par > 0:
    t_real = t_par
elif isinstance(t_rampa, float) and t_rampa > 0:
    t_real = t_rampa

st.markdown("---")
if isinstance(t_real, float) and t_real > 0:
    st.success(f"**Tiempo gobernante estimado (25→50 Hz)**: {fmt(t_real, 's')}")
else:
    st.warning("Faltan datos para calcular el tiempo gobernante. Revisa \(J_{eq}\), \(T_{nom}\) y relación \(r\).")

st.caption("Próximo paso: integración hidráulica con \(Q(n)\), \(H(Q)\) y \(η(Q)\).")
