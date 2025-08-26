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


# Mapeo 1:1 contra dataset.csv (sep=';' decimal=',')
# type: "num" → convertir con get_num(); "str" → usar tal cual.
ATTR = {
    # ── Identificación / transmisión ─────────────────────────────────────────
    "TAG":                 {"col": "TAG",                  "unit": "",        "type": "str"},
    "r":                   {"col": "r_trans",              "unit": "",        "type": "num"},   # r = n_motor / n_bomba
    "series":              {"col": "series",               "unit": "",        "type": "str"},
    "grooves":             {"col": "grooves",              "unit": "",        "type": "str"},
    "centerdistance_mm":   {"col": "centerdistance_mm",    "unit": "mm",      "type": "num"},

    # ── Geometría poleas / ejes (catálogo TB Woods) ─────────────────────────
    "driver_od_in":        {"col": "driver_od_in",         "unit": "in",      "type": "num"},
    "driver_bushing":      {"col": "driver_bushing",       "unit": "",        "type": "str"},
    "driver_shaft_mm":     {"col": "driver_shaft_mm",      "unit": "mm",      "type": "num"},
    "driven_od_in":        {"col": "driven_od_in",         "unit": "in",      "type": "num"},
    "driven_weight_lb":    {"col": "driven_weight_lb",     "unit": "lb",      "type": "num"},
    "driven_bushing":      {"col": "driven_bushing",       "unit": "",        "type": "str"},
    "driven_shaft_mm":     {"col": "driven_shaft_mm",      "unit": "mm",      "type": "num"},

    # ── Bomba / motor ───────────────────────────────────────────────────────
    "pump_model":          {"col": "pumpmodel",            "unit": "",        "type": "str"},
    "motorpower_kw":       {"col": "motorpower_kw",        "unit": "kW",      "type": "num"},
    "poles":               {"col": "poles",                "unit": "",        "type": "num"},
    "t_nom_nm":            {"col": "t_nom_nm",             "unit": "N·m",     "type": "num"},
    "J_m":                 {"col": "motor_j_kgm2",         "unit": "kg·m²",   "type": "num"},

    # ── Impulsor (Metso) ────────────────────────────────────────────────────
    "impeller_d_mm":       {"col": "impeller_d_mm",        "unit": "mm",      "type": "num"},
    "impeller_mass_kg":    {"col": "impeller_mass_kg",     "unit": "kg",      "type": "num"},
    "J_imp":               {"col": "impeller_j_kgm2",      "unit": "kg·m²",   "type": "num"},

    # ── Velocidades (25–50 Hz) ──────────────────────────────────────────────
    "n_m_min":             {"col": "motor_n_min_rpm",      "unit": "rpm",     "type": "num"},
    "n_m_max":             {"col": "motor_n_max_rpm",      "unit": "rpm",     "type": "num"},
    "n_p_min":             {"col": "pump_n_min_rpm",       "unit": "rpm",     "type": "num"},
    "n_p_max":             {"col": "pump_n_max_rpm",       "unit": "rpm",     "type": "num"},
    "n_ref_rpm":           {"col": "n_ref_rpm",            "unit": "rpm",     "type": "num"},

    # ── Poleas y BUSHINGS (TB Woods) ────────────────────────────────────────
    "driverpulley_weight_kg": {"col": "driverpulley_weight_kg", "unit": "kg",   "type": "num"},
    "J_driver":            {"col": "driverpulley_j_kgm2",  "unit": "kg·m²",   "type": "num"},
    # Nota: claves "Bushing" y "Sleeve" referencian la MISMA columna para compatibilidad
    "J_bushing_driver":    {"col": "driverbushing_j_kgm2", "unit": "kg·m²",   "type": "num"},
    "J_sleeve_driver":     {"col": "driverbushing_j_kgm2", "unit": "kg·m²",   "type": "num"},
    "J_driven":            {"col": "drivenpulley_j_kgm2",  "unit": "kg·m²",   "type": "num"},
    "J_bushing_driven":    {"col": "drivenbushing_j_kgm2", "unit": "kg·m²",   "type": "num"},
    "J_sleeve_driven":     {"col": "drivenbushing_j_kgm2", "unit": "kg·m²",   "type": "num"},

    # ── Curva del sistema H(Q)=H0+K(Q/3600)^2 ───────────────────────────────
    "H0_m":                {"col": "H0_m",                 "unit": "m",       "type": "num"},
    "K_m_s2":              {"col": "K_m_s2",               "unit": "",        "type": "num"},
    "R2_H":                {"col": "R2_H",                 "unit": "",        "type": "num"},

    # ── Eficiencia η(Q)=η_a+η_bβ+η_cβ² (β=Q/Q_ref) ─────────────────────────
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

    # ── Densidades ──────────────────────────────────────────────────────────
    "rho_kgm3":            {"col": "rho_kgm3",             "unit": "kg/m³",   "type": "num"},   # genérica/agua
    "SlurryDensity":       {"col": "SlurryDensity_kgm3",   "unit": "kg/m³",   "type": "num"},   # pulpa (para sección 4)
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
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return str(v)


# =============================================================================
# Cargar dataset + UI (logos y título)
# =============================================================================
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

# Atributos base
txt_pump_model = get_attr(row, "pump_model")
P_motor_kW     = get_attr(row, "motorpower_kw")
T_nom_nm       = get_attr(row, "t_nom_nm")
r_nm_np        = get_attr(row, "r")
D_imp_mm       = get_attr(row, "impeller_d_mm")

n_m_min = get_attr(row, "n_m_min")
n_m_max = get_attr(row, "n_m_max")
n_p_min = get_attr(row, "n_p_min")
n_p_max = get_attr(row, "n_p_max")

# Hidráulica y eficiencia
H0_m     = get_attr(row, "H0_m")
K_ms2    = get_attr(row, "K_m_s2")
R2_H     = get_attr(row, "R2_H")
Q_min_ds = get_attr(row, "Qmin_m3h")
Q_ref    = get_attr(row, "Q_ref_m3h")
Q_max_ds = get_attr(row, "Qmax_m3h")
rho      = get_attr(row, "SlurryDensity")

eta_a    = get_attr(row, "eta_a")
eta_b    = get_attr(row, "eta_b")
eta_c    = get_attr(row, "eta_c")
R2_eta   = get_attr(row, "R2_eta")
eta_fbk  = get_attr(row, "eta")             # fallback si faltan coeficientes

# Fallbacks razonables
if np.isnan(rho) or rho <= 0:
    rho = 1000.0
if np.isnan(Q_ref) or Q_ref <= 0:
    qb = get_attr(row, "Qbest_m3h")
    if not np.isnan(qb) and qb > 0:
        Q_ref = qb
    elif not (np.isnan(Q_min_ds) or np.isnan(Q_max_ds)):
        Q_ref = 0.5 * (Q_min_ds + Q_max_ds)
    else:
        Q_ref = 300.0

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
    st.markdown(
        f"- Relación transmisión $\\big(r=\\tfrac{{n_{{motor}}}}{{n_{{bomba}}}}\\big)$: {val_blue(r_nm_np, '')}",
        unsafe_allow_html=True,
    )
    st.markdown(f"- Velocidad motor min–max: {val_blue(n_m_min, 'rpm', 0)} – {val_blue(n_m_max, 'rpm', 0)}", unsafe_allow_html=True)
with c3:
    st.markdown("**Bomba (25–50 Hz por afinidad)**")
    st.markdown(f"- Velocidad bomba min–max: {val_blue(n_p_min, 'rpm', 0)} – {val_blue(n_p_max, 'rpm', 0)}", unsafe_allow_html=True)
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
    J_bushing_driver = get_attr(row, "J_sleeve_driver")
    if np.isnan(J_bushing_driver):
        J_bushing_driver = 0.10 * (J_driver if not np.isnan(J_driver) else 0.0)
    J_driven = get_attr(row, "J_driven")
    J_bushing_driven = get_attr(row, "J_sleeve_driven")
    if np.isnan(J_bushing_driven):
        J_bushing_driven = 0.10 * (J_driven if not np.isnan(J_driven) else 0.0)
    J_imp = get_attr(row, "J_imp")
    r     = max(get_attr(row, "r"), 1e-6)

    st.markdown(f"- Motor (J_m): {val_blue(J_m, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Polea motriz (J_driver): {val_blue(J_driver, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Bushing motriz (J_bushing_driver≈10% J_driver): {val_blue(J_bushing_driver, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Polea conducida (J_driven): {val_blue(J_driven, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Bushing conducido (J_bushing_driven≈10% J_driven): {val_blue(J_bushing_driven, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Impulsor/rotor de bomba (J_imp): {val_blue(J_imp, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(
        f"- Relación $r=\\tfrac{{n_m}}{{n_p}}$: {val_blue(r, '')}",
        unsafe_allow_html=True,
    )

    J_eq = (J_m + J_driver + J_bushing_driver) + (J_driven + J_bushing_driven + J_imp) / (r**2)
    st.markdown(f"**Inercia equivalente (J_eq):** {val_green(J_eq, 'kg·m²')}", unsafe_allow_html=True)

with colR:
    st.subheader("Fórmula utilizada")
    st.latex(r"J_{\mathrm{eq}} = J_m + J_{\mathrm{driver}} + J_{\mathrm{bushing,driver}} + \dfrac{J_{\mathrm{driven}} + J_{\mathrm{bushing,driven}} + J_{\mathrm{imp}}}{r^2}")
    with st.expander("Formulación de las inercias por componente", expanded=True):
        st.markdown(
            "- **Poleas** (J_driver, J_driven): catálogo **TB Woods**.\n"
            "- **Bushings**: si falta dato, **10%** de la inercia de su polea.\n"
            "- **Impulsor** (J_imp): manuales **Metso**."
        )
        st.markdown("Las inercias del lado bomba giran a:")
        st.latex(r"\omega_p=\dfrac{\omega_m}{r}")
        st.markdown("Igualando energías cinéticas a una $ \omega_m $ común, los términos del lado de la bomba se dividen por $ r^2 $:")
        st.latex(r"E=\tfrac12\Big[J_m+J_{\mathrm{driver}}+J_{\mathrm{bushing,driver}}+\tfrac{J_{\mathrm{driven}}+J_{\mathrm{bushing,driven}}+J_{\mathrm{imp}}}{r^2}\Big]\omega_m^2")

st.markdown("---")


# =============================================================================
# 3) Tiempo inercial (par disponible vs rampa)
# =============================================================================
st.markdown("## 3) Tiempo inercial (par disponible vs rampa VDF)")
# Aumentar rango de rampa a 600 rpm/s
rampa_vdf = st.slider("Rampa VDF en el motor [rpm/s]", min_value=10, max_value=600, value=100, step=5)

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

# Tiempo limitante (evitar ** en HTML)
lim3_txt = "Tiempo limitante (sección 3): "
if not np.isnan(t_par) and t_par > t_rampa:
    pill(lim3_txt + f"por par = {fmt_num(t_par, 's')}")
    limit3_name, limit3_val = "par", t_par
else:
    pill(lim3_txt + f"por rampa VDF = {fmt_num(t_rampa, 's')}")
    limit3_name, limit3_val = "rampa VDF", t_rampa

with st.expander("Detalles y fórmulas — Sección 3", expanded=False):
    st.markdown("**Hipótesis:** par del motor constante y pérdidas mecánicas despreciables en 25–50 Hz.")
    st.latex(r"T_{\mathrm{disp}}=T_{\mathrm{nom}}")
    st.markdown("**Dinámica rotacional (eje motor):**")
    st.latex(r"J_{eq}\,\dot{\omega}_m = T_{\mathrm{disp}} \;\Rightarrow\; \dot{\omega}_m=\frac{T_{\mathrm{disp}}}{J_{eq}}")
    st.markdown("**Conversión a rpm:**")
    st.latex(r"n_m=\frac{60}{2\pi}\,\omega_m \;\Rightarrow\; \dot n_m=\frac{60}{2\pi}\,\frac{T_{\mathrm{disp}}}{J_{eq}}")
    st.markdown("**Tiempos:**")
    st.latex(r"t_{\mathrm{par}}=\frac{\Delta n_m}{\dot n_m},\qquad t_{\mathrm{rampa}}=\frac{\Delta n_m}{\text{rampa}}")
    st.markdown("**Criterio:**")
    st.latex(r"t=\max\{t_{\mathrm{par}},\,t_{\mathrm{rampa}}\}")

st.markdown("---")


# =============================================================================
# 4) Integración con carga hidráulica
# =============================================================================
st.markdown("## 4) Integración con carga hidráulica")

# Fórmulas iniciales
st.latex(r"H(Q)=H_0+K\left(\dfrac{Q}{3600}\right)^2 \qquad \big[Q:\ \mathrm{m^3/h},\ H:\ \mathrm{m}\big]")
st.latex(r"\eta(Q)=\eta_a+\eta_b\,\beta+\eta_c\,\beta^2,\qquad \beta=\dfrac{Q}{Q_{\mathrm{ref}}},\quad \eta\in[0.40,\,0.88]")
st.latex(r"\dfrac{dt}{dn_m}=\dfrac{J_{eq}\,(2\pi/60)}{T_{\mathrm{disp}}-T_{\mathrm{load},m}(n_m)}\quad\Rightarrow\quad t=\int_{n_{m,\min}}^{n_{m,\max}}\dfrac{J_{eq}\,(2\pi/60)}{T_{\mathrm{disp}}-T_{\mathrm{load},m}(n_m)}\,dn_m")

# Mostrar parámetros hidráulicos del TAG (incluye R²)
c4a, c4b, c4c, c4d, c4e = st.columns(5)
with c4a:
    st.markdown(f"- H₀: {val_blue(H0_m, 'm')}", unsafe_allow_html=True)
with c4b:
    st.markdown(f"- K (K_m_s2): {val_blue(K_ms2, '', 3)}", unsafe_allow_html=True)
with c4c:
    st.markdown(f"- ρ: {val_blue(rho, 'kg/m³', 0)}", unsafe_allow_html=True)
with c4d:
    st.markdown(f"- R² curva H: {val_blue(R2_H, '', 3)}", unsafe_allow_html=True)
with c4e:
    st.markdown(f"- R² eficiencia: {val_blue(R2_eta, '', 3)}", unsafe_allow_html=True)

# ----- Rango de caudales por TAG (±30% respecto a Qmin/Qmax del dataset) -----
def _qmin_qmax_from_tag():
    """Devuelve (qmin_base, qmax_base) y (qmin_allowed, qmax_allowed) por TAG."""
    if not np.isnan(Q_min_ds) and not np.isnan(Q_max_ds) and Q_min_ds < Q_max_ds:
        qmin_base = float(Q_min_ds)
        qmax_base = float(Q_max_ds)
    else:
        # Fallback si el TAG no trae rango
        qmin_base, qmax_base = 100.0, 300.0

    qmin_allowed = max(0.0, 0.70 * qmin_base)
    qmax_allowed = max(qmin_allowed + 1.0, 1.30 * qmax_base)
    return qmin_base, qmax_base, qmin_allowed, qmax_allowed

qmin_base, qmax_base, qmin_allowed, qmax_allowed = _qmin_qmax_from_tag()

# Clave única por TAG para forzar re-creación del slider al cambiar de TAG
slider_key = f"q_slider_{tag_sel}"

Q_min, Q_max = st.slider(
    "Rango de caudal considerado [m³/h]  (límite: −30% desde Q_min y +30% desde Q_max del TAG)",
    min_value=float(qmin_allowed),
    max_value=float(qmax_allowed),
    value=(float(qmin_base), float(qmax_base)),
    step=1.0,
    key=slider_key,
)

# Mapear velocidad bomba → caudal (lineal 25–50 Hz)
def Q_from_np(n_p: np.ndarray) -> np.ndarray:
    if n_p_max <= n_p_min + 1e-9:
        return np.full_like(n_p, Q_min)
    return Q_min + (Q_max - Q_min) * (n_p - n_p_min) / (n_p_max - n_p_min)

# Eficiencia por polinomio (o fallback)
def eta_from_Q(Q_m3h: np.ndarray) -> np.ndarray:
    if not (np.isnan(eta_a) or np.isnan(eta_b) or np.isnan(eta_c) or np.isnan(Q_ref) or Q_ref <= 0):
        beta = Q_m3h / Q_ref
        e = eta_a + eta_b * beta + eta_c * (beta**2)
    else:
        base = 0.72 if np.isnan(eta_fbk) else float(eta_fbk)
        e = np.full_like(Q_m3h, base)
    return np.clip(e, 0.40, 0.88)

# Mallas en velocidad del motor/bomba
N = 600
n_m_grid = np.linspace(n_m_min, n_m_max, N)
n_p_grid = n_m_grid / max(r_nm_np, 1e-9)

Q_grid   = Q_from_np(n_p_grid)          # m³/h
q_grid   = Q_grid / 3600.0              # m³/s
H_grid   = H0_m + K_ms2 * (q_grid**2)   # m
eta_grid = eta_from_Q(Q_grid)           # —

# Potencias y pares
g = 9.81
P_h_grid = rho * g * q_grid * H_grid / np.maximum(eta_grid, 1e-6)   # W
omega_p  = 2.0 * math.pi * n_p_grid / 60.0                          # rad/s
T_pump   = np.where(omega_p > 1e-9, P_h_grid / omega_p, 0.0)        # N·m (eje bomba)
T_load_m = T_pump / max(r_nm_np, 1e-9)                              # N·m (eje motor)
T_disp_m = np.full_like(T_load_m, T_nom_nm)                         # N·m (motor)
T_net    = T_disp_m - T_load_m                                      # N·m (motor)

# Gráfico: Par resistente vs Par disponible (sin línea 0 N·m)
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

# Gráfico: Par neto (mantiene línea 0 N·m, ahora gris)
fig_net = go.Figure()
fig_net.add_trace(go.Scatter(x=n_p_grid, y=T_net, mode="lines", name="Par neto T_net (motor)"))
fig_net.add_trace(go.Scatter(x=n_p_grid, y=np.zeros_like(n_p_grid), mode="lines",
                             name="0 N·m", line=dict(dash="dot", color="#888")))
fig_net.update_layout(
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis_title="Velocidad bomba n_p [rpm]",
    yaxis_title="Par neto en motor [N·m]",
    legend=dict(orientation="h"),
    height=320,
)
st.plotly_chart(fig_net, use_container_width=True)

# Gráfico: Integrando dt/dn_m (área ≈ tiempo)
T_net_clip = np.maximum(T_net, 1e-6)  # evitar división por 0
integrand = (J_eq * (2.0 * math.pi / 60.0)) / T_net_clip  # [s/rpm]
integrand_pos = np.where(T_net > 1e-9, integrand, np.nan)

fig_area = go.Figure()
fig_area.add_trace(go.Scatter(
    x=n_m_grid, y=integrand_pos,
    mode="lines", name="dt/dn_m [s/rpm]", fill="tozeroy"
))
fig_area.update_layout(
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis_title="Velocidad motor n_m [rpm]",
    yaxis_title="Integrando dt/dn_m [s/rpm] (área ≈ tiempo)",
    legend=dict(orientation="h"),
    height=320,
)
st.plotly_chart(fig_area, use_container_width=True)

# Integración temporal con carga hidráulica
if not (np.isnan(J_eq) or J_eq <= 0):
    omega_m_grid = 2.0 * math.pi * n_m_grid / 60.0
    d_omega = np.diff(omega_m_grid)
    dt = (J_eq * d_omega) / T_net_clip[:-1]                 # s
    t_hyd = float(np.sum(dt))
else:
    t_hyd = float("nan")

# Mostrar tiempo hidráulico y, JUSTO DEBAJO, el tiempo limitante (sección 4)
c_time = st.container()
with c_time:
    st.markdown(f"- Tiempo por carga **hidráulica** (integración): {val_green(t_hyd, 's')}", unsafe_allow_html=True)
    t_rampa_local = (n_m_max - n_m_min) / max(rampa_vdf, 1e-9)
    t_lim_4 = max(t_hyd, t_rampa_local) if not np.isnan(t_hyd) else t_rampa_local
    which = "hidráulica" if (not np.isnan(t_hyd) and t_hyd > t_rampa_local) else "rampa VDF"
    pill(f"Tiempo limitante (sección 4): {which} = {fmt_num(t_lim_4, 's')}")

with st.expander("Detalles y fórmulas — Sección 4", expanded=False):
    st.markdown("**Curva del sistema (K=K_m_s2 del dataset):**")
    st.latex(r"H(Q)=H_0+K\left(\dfrac{Q}{3600}\right)^2")
    st.markdown("**Afinidad (25–50 Hz):** interpolación lineal con fórmula:")
    st.latex(r"Q \propto n_p,\qquad Q(n_p)=Q_{\min}+\big(Q_{\max}-Q_{\min}\big)\,\dfrac{n_p-n_{p,\min}}{n_{p,\max}-n_{p,\min}}")
    st.markdown("**Eficiencia:** si hay coeficientes se usa polinomio en $\\beta$; si no, valor del dataset acotado a $[0.40,0.88]$.")
    st.latex(r"\beta=\dfrac{Q}{Q_{\mathrm{ref}}},\qquad \eta(Q)=\eta_a+\eta_b\,\beta+\eta_c\,\beta^2,\qquad \eta\in[0.40,0.88]")
    st.markdown("**Potencia hidráulica:**")
    st.latex(r"P_h=\dfrac{\rho g\,Q_s\,H(Q)}{\eta(Q)},\qquad Q_s=\dfrac{Q}{3600}")
    st.markdown("**Par y reflejo al motor:**")
    st.latex(r"T_{\mathrm{pump}}=\dfrac{P_h}{\omega_p},\quad \omega_p=\dfrac{2\pi n_p}{60},\qquad T_{\mathrm{load},m}=\dfrac{T_{\mathrm{pump}}}{r}")
    st.markdown("**Par neto y tiempo integrado:**")
    st.latex(r"T_{\mathrm{net}}=T_{\mathrm{disp}}-T_{\mathrm{load},m},\qquad \Delta t=\dfrac{J_{eq}\,\Delta\omega_m}{T_{\mathrm{net}}}")
    st.markdown("**Tiempo total:**")
    st.latex(r"t=\int_{n_{m,\min}}^{n_{m,\max}}\frac{dt}{dn_m}\,dn_m\;=\;\int_{n_{m,\min}}^{n_{m,\max}}\dfrac{J_{eq}\,(2\pi/60)}{T_{\mathrm{net}}(n_m)}\,dn_m")

st.markdown("---")


# =============================================================================
# 5) Exportar resultados (CSV)
# =============================================================================
st.markdown("## 5) Exportar resultados (CSV)")

# Para exportación – determinar tiempo limitante global
def _safe(v):
    return -np.inf if np.isnan(v) else v

candidates = [("par", t_par), ("rampa VDF", (n_m_max - n_m_min) / max(rampa_vdf, 1e-9)), ("hidráulica", t_hyd)]
limit_global_name, limit_global_val = max(candidates, key=lambda kv: _safe(kv[1]))

summary = {
    "TAG": tag_sel,
    "pump_model": txt_pump_model,
    "impeller_d_mm": D_imp_mm,
    "motorpower_kw": P_motor_kW,
    "t_nom_nm": T_nom_nm,
    "r_nm_np": r_nm_np,
    "n_m_min_rpm": n_m_min,
    "n_m_max_rpm": n_m_max,
    "n_p_min_rpm": n_p_min,
    "n_p_max_rpm": n_p_max,
    "J_m_kgm2": get_attr(row, "J_m"),
    "J_driver_kgm2": get_attr(row, "J_driver"),
    "J_bushing_driver_used_kgm2": get_attr(row, "J_sleeve_driver") if not np.isnan(get_attr(row, "J_sleeve_driver")) else 0.10 * (get_attr(row, "J_driver") if not np.isnan(get_attr(row, "J_driver")) else 0.0),
    "J_driven_kgm2": get_attr(row, "J_driven"),
    "J_bushing_driven_used_kgm2": get_attr(row, "J_sleeve_driven") if not np.isnan(get_attr(row, "J_sleeve_driven")) else 0.10 * (get_attr(row, "J_driven") if not np.isnan(get_attr(row, "J_driven")) else 0.0),
    "J_imp_kgm2": get_attr(row, "J_imp"),
    "J_eq_kgm2": J_eq,
    "H0_m": H0_m,
    "K_m_s2": K_ms2,
    "R2_H": R2_H,
    "eta_a": eta_a,
    "eta_b": eta_b,
    "eta_c": eta_c,
    "R2_eta": R2_eta,
    "Q_ref_m3h": Q_ref,
    "rho_kgm3": rho,
    "Q_slider_min_m3h": Q_min,
    "Q_slider_max_m3h": Q_max,
    "rampa_vdf_rpmps": rampa_vdf,
    "n_dot_torque_rpms": n_dot_torque,
    "t_par_s": t_par,
    "t_rampa_s": (n_m_max - n_m_min) / max(rampa_vdf, 1e-9),
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
