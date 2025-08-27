# app.py
# ─────────────────────────────────────────────────────────────────────────────
# Memoria de Cálculo — Tiempo de reacción (VDF) con integración hidráulica
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# Config general
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Memoria de Cálculo – Tiempo de reacción (VDF)", layout="wide")
BLUE = "#1f77b4"   # Dado (dataset)
GREEN = "#2ca02c"  # Calculado

# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------
def dataset_path() -> Path:
    return Path(__file__).with_name("dataset.csv")

def image_file(name: str) -> str | None:
    p1 = Path(__file__).with_name("images") / name
    p2 = Path("images") / name
    if p1.exists(): return str(p1)
    if p2.exists(): return str(p2)
    return None

def get_num(x) -> float:
    if x is None: return float("nan")
    if isinstance(x, (int, float, np.number)): return float(x)
    s = str(x).strip().replace(" ", "").replace("\u00a0", "")
    s = s.replace(",", ".")
    import re
    s = re.sub(r"[^0-9eE\+\-\.]", "", s)
    try: return float(s)
    except Exception: return float("nan")

def fmt_num(x, unit: str = "", ndigits: int = 2) -> str:
    if isinstance(x, str): return f"{x} {unit}".strip()
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))): return "—"
    s = f"{x:,.{ndigits}f}".replace(",", "_").replace(".", ",").replace("_", ".")
    return f"{s} {unit}".strip()

def color_value(text: str, color: str = BLUE, bold: bool = True) -> str:
    w = "600" if bold else "400"
    return f'<span style="color:{color}; font-weight:{w}">{text}</span>'

def val_blue(x, unit: str = "", ndigits: int = 2) -> str:
    if isinstance(x, str): return color_value(x, BLUE)
    return color_value(fmt_num(x, unit, ndigits), BLUE)

def val_green(x, unit: str = "", ndigits: int = 2) -> str:
    if isinstance(x, str): return color_value(x, GREEN)
    return color_value(fmt_num(x, unit, ndigits), GREEN)

def pill(text: str, bg: str = "#e8f5e9", color: str = "#1b5e20"):
    st.markdown(
        f"""
        <div style="border-left: 5px solid {color}; background:{bg};
                    padding:0.8rem 1rem; border-radius:0.5rem; margin-top:0.5rem">
            <span style="color:{color}; font-weight:600">{text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Carga de datos
# -----------------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(dataset_path(), sep=";", decimal=",")
    df.columns = [c.strip() for c in df.columns]
    return df

df = load_data()

# Mapeo (según columnas que indicaste)
COL = {
    "TAG": "TAG",
    "r": "r_trans",
    "pump_model": "pumpmodel",
    "motorpower_kw": "motorpower_kw",
    "t_nom_nm": "t_nom_nm",
    "J_m": "motor_j_kgm2",
    "J_driver": "driverpulley_j_kgm2",
    "J_bushing_driver": "driverbushing_j_kgm2",
    "J_driven": "drivenpulley_j_kgm2",
    "J_bushing_driven": "drivenbushing_j_kgm2",
    "J_imp": "impeller_j_kgm2",
    "impeller_d_mm": "impeller_d_mm",
    "n_m_min": "motor_n_min_rpm",
    "n_m_max": "motor_n_max_rpm",
    "n_p_min": "pump_n_min_rpm",
    "n_p_max": "pump_n_max_rpm",
    "H0_m": "H0_m",
    "K_m_s2": "K_m_s2",
    "eta_a": "eta_a",
    "eta_b": "eta_b",
    "eta_c": "eta_c",
    "R2_eta": "R2_eta",
    "Q_min_m3h": "Q_min_m3h",
    "Q_max_m3h": "Q_max_m3h",
    "Q_ref_m3h": "Q_ref_m3h",
    "n_ref_rpm": "n_ref_rpm",
    "rho_kgm3": "rho_kgm3",
    "eta_beta": "eta_beta",
    "eta_min": "eta_min_clip",
    "eta_max": "eta_max_clip",
    "SlurryDensity": "SlurryDensity_kgm3",
}

def v(row, key, default=np.nan):
    c = COL[key]
    if c not in row or pd.isna(row[c]): return default
    return get_num(row[c]) if isinstance(row[c], (int, float, np.number, str)) else row[c]

# -----------------------------------------------------------------------------
# Encabezado + Notas
# -----------------------------------------------------------------------------
cL, cC, cR = st.columns([1, 3, 1])
with cL:
    f = image_file("metso_logo.png")
    if f: st.image(f, use_container_width=True)
with cC:
    st.markdown("<h1 style='text-align:center;margin:0.2rem 0'>Tiempo de reacción de bombas con VDF</h1>", unsafe_allow_html=True)
with cR:
    f = image_file("ausenco_logo.png")
    if f: st.image(f, use_container_width=True)
st.markdown("---")

with st.expander("Notas clave de esta versión", expanded=True):
    st.markdown(
        "- **Slider de la Sección 4** en **rpm (25–50 Hz)** y el caudal se obtiene por **afinidad estricta** $Q=k\,n_p$ (sin offsets).\n"
        "- **Selector de modelo de eficiencia**: polinómica, constante promedio y constante fija por usuario. Se muestran **η mínima, máxima y media** efectivamente usadas.\n"
        "- Si en algún punto $T_{\\rm net}\\le 0$ no se integra y se informa **par insuficiente**.\n"
        "- Gráfico de **$dt/dn_m$** con **área integrada**.\n"
        "- Estilo: **datos en azul**, **cálculos en verde** y **fórmulas en LaTeX**.\n"
        "- **Sección 5** exporta todos los resultados del TAG a **CSV**."
    )

# -----------------------------------------------------------------------------
# Selector de TAG
# -----------------------------------------------------------------------------
tag = st.sidebar.selectbox("Selecciona TAG", df[COL["TAG"]].astype(str).tolist(), index=0)
row = df[df[COL["TAG"]].astype(str) == str(tag)].iloc[0]

# -----------------------------------------------------------------------------
# Lectura de parámetros por TAG
# -----------------------------------------------------------------------------
pump_model = str(row.get(COL["pump_model"], "—"))
P_motor_kW = v(row, "motorpower_kw")
T_nom_nm    = v(row, "t_nom_nm")
r_nm_np     = v(row, "r")

J_m   = v(row, "J_m", 0.0)
J_dr  = v(row, "J_driver", 0.0)
J_bdr = v(row, "J_bushing_driver", np.nan)
if np.isnan(J_bdr): J_bdr = 0.10 * J_dr
J_dn  = v(row, "J_driven", 0.0)
J_bdn = v(row, "J_bushing_driven", np.nan)
if np.isnan(J_bdn): J_bdn = 0.10 * J_dn
J_imp = v(row, "J_imp", 0.0)

D_imp_mm = v(row, "impeller_d_mm")

n_m_min = v(row, "n_m_min")
n_m_max = v(row, "n_m_max")
n_p_min = v(row, "n_p_min")
n_p_max = v(row, "n_p_max")

H0_m   = v(row, "H0_m")
K_m_s2 = v(row, "K_m_s2")
eta_a  = v(row, "eta_a")
eta_b  = v(row, "eta_b")
eta_c  = v(row, "eta_c")
eta_min = v(row, "eta_min", 0.40)
eta_max = v(row, "eta_max", 0.88)
eta_beta = v(row, "eta_beta")
Q_min_ds = v(row, "Q_min_m3h")
Q_max_ds = v(row, "Q_max_m3h")
Q_ref    = v(row, "Q_ref_m3h")
n_ref_rpm = v(row, "n_ref_rpm")
rho      = v(row, "rho_kgm3", 1000.0)
g = 9.81

# -----------------------------------------------------------------------------
# 1) Parámetros
# -----------------------------------------------------------------------------
st.markdown("## 1) Parámetros")
c1, c2, c3 = st.columns([1.2, 1.2, 1.2])

with c1:
    st.markdown("**Identificación**")
    st.markdown(f"- Modelo de bomba: {val_blue(pump_model, '', 0)}", unsafe_allow_html=True)
    st.markdown(f"- TAG: {val_blue(tag, '', 0)}", unsafe_allow_html=True)
    st.markdown(f"- Diámetro impulsor: {val_blue(D_imp_mm, 'mm')}", unsafe_allow_html=True)

with c2:
    st.markdown("**Motor & transmisión**")
    st.markdown(f"- Potencia motor instalada: {val_blue(P_motor_kW, 'kW')}", unsafe_allow_html=True)
    st.markdown(f"- Par nominal del motor: {val_blue(T_nom_nm, 'N·m')}", unsafe_allow_html=True)
    st.markdown("Relación transmisión (fórmula):")
    st.latex(r"r=\frac{n_{\mathrm{motor}}}{n_{\mathrm{bomba}}}")
    st.markdown(f"Valor: {val_blue(r_nm_np)}", unsafe_allow_html=True)
    st.markdown(f"- Velocidad motor min–max: {val_blue(n_m_min, 'rpm', 0)} – {val_blue(n_m_max, 'rpm', 0)}", unsafe_allow_html=True)

with c3:
    st.markdown("**Bomba (25–50 Hz)**")
    st.markdown(f"- Velocidad bomba min–max: {val_blue(n_p_min, 'rpm', 0)} – {val_blue(n_p_max, 'rpm', 0)}", unsafe_allow_html=True)
    st.markdown(f"- Densidad de pulpa ρ: {val_blue(rho, 'kg/m³', 0)}", unsafe_allow_html=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# 2) Cálculo de inercia equivalente
# -----------------------------------------------------------------------------
st.header("2) Cálculo de inercia equivalente")
cL2, cR2 = st.columns([1.1, 1])

with cL2:
    st.subheader("Inercias individuales")
    st.markdown(f"- Motor (J_m): {val_blue(J_m, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Polea motriz (J_driver): {val_blue(J_dr, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Bushing motriz (≈10% J_driver): {val_blue(J_bdr, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Polea conducida (J_driven): {val_blue(J_dn, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Bushing conducido (≈10% J_driven): {val_blue(J_bdn, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Impulsor/rotor de bomba (J_imp): {val_blue(J_imp, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Relación r (n_m/n_p): {val_blue(r_nm_np)}", unsafe_allow_html=True)

    r_eff = max(r_nm_np, 1e-9)
    J_eq = (J_m + J_dr + J_bdr) + (J_dn + J_bdn + J_imp) / (r_eff**2)
    st.markdown(f"**Inercia equivalente (J_eq):** {val_green(J_eq, 'kg·m²')}", unsafe_allow_html=True)

with cR2:
    st.subheader("Fórmula utilizada")
    st.latex(
        r"J_{\mathrm{eq}} = J_m + J_{\mathrm{driver}} + J_{\mathrm{bushing,driver}}"
        r"+ \frac{J_{\mathrm{driven}} + J_{\mathrm{bushing,driven}} + J_{\mathrm{imp}}}{r^2}"
    )
    with st.expander("Formulación de las inercias por componente", expanded=True):
        st.markdown(
            "- **Poleas**: catálogo **TB Woods**.\n"
            "- **Bushing**: si falta dato, **10%** de la inercia de su polea.\n"
            "- **Impulsor**: manuales **Metso**."
        )
        st.latex(r"\omega_p=\frac{\omega_m}{r}")
        st.latex(
            r"\tfrac{1}{2}J_m\omega_m^2 + \tfrac{1}{2}J_{\mathrm{driver}}\omega_m^2 "
            r"+ \tfrac{1}{2}J_{\mathrm{bushing,driver}}\omega_m^2 "
            r"+ \tfrac{1}{2}J_{\mathrm{driven}}\!\left(\tfrac{\omega_m}{r}\right)^{\!2} "
            r"+ \tfrac{1}{2}J_{\mathrm{bushing,driven}}\!\left(\tfrac{\omega_m}{r}\right)^{\!2} "
            r"+ \tfrac{1}{2}J_{\mathrm{imp}}\!\left(\tfrac{\omega_m}{r}\right)^{\!2} "
            r"= \tfrac{1}{2}J_{\mathrm{eq}}\omega_m^2"
        )

with st.expander("Descripción — Sección 2"):
    st.markdown("La **inercia equivalente** reúne todas las masas rotantes vistas desde el **eje motor**.")

st.markdown("---")

# -----------------------------------------------------------------------------
# 3) Tiempo inercial (par disponible vs rampa VDF)
# -----------------------------------------------------------------------------
st.markdown("## 3) Tiempo inercial (par disponible vs rampa VDF)")
rampa_vdf = st.slider("Rampa VDF en el motor [rpm/s]", min_value=10, max_value=600, value=100, step=5)

if not (np.isnan(J_eq) or J_eq <= 0 or np.isnan(T_nom_nm)):
    n_dot_torque = (60.0 / (2.0 * math.pi)) * (T_nom_nm / J_eq)  # rpm/s
    t_par = (n_m_max - n_m_min) / max(n_dot_torque, 1e-9)
else:
    n_dot_torque = float("nan")
    t_par = float("nan")

t_rampa = (n_m_max - n_m_min) / max(rampa_vdf, 1e-9)

cA, cB, cC = st.columns(3)
with cA: st.markdown(f"- Aceleración por par (lado motor): {val_green(n_dot_torque, 'rpm/s')}", unsafe_allow_html=True)
with cB: st.markdown(f"- Tiempo por par (25→50 Hz): {val_green(t_par, 's')}", unsafe_allow_html=True)
with cC: st.markdown(f"- Tiempo por rampa VDF: {val_blue(t_rampa, 's')}", unsafe_allow_html=True)

lim3 = "Tiempo limitante (sección 3): "
pill(lim3 + (f"<b>por par</b> = {fmt_num(t_par, 's')}" if (not np.isnan(t_par) and t_par > t_rampa)
             else f"<b>por rampa VDF</b> = {fmt_num(t_rampa, 's')}"))

with st.expander("Detalles y fórmulas — Sección 3"):
    st.markdown("**Hipótesis**")
    st.latex(r"T_{\mathrm{disp}}=T_{\mathrm{nom}} \quad \text{(constante en 25\text{–}50 Hz)}")
    st.markdown("**Dinámica rotacional**")
    st.latex(r"J_{eq}\,\dot{\omega}_m=T_{\mathrm{disp}} \;\Rightarrow\; \dot{\omega}_m=\frac{T_{\mathrm{disp}}}{J_{eq}}")
    st.markdown("**Conversión a rpm**")
    st.latex(r"n_m=\tfrac{60}{2\pi}\,\omega_m \;\Rightarrow\; \dot n_m=\tfrac{60}{2\pi}\,\tfrac{T_{\mathrm{disp}}}{J_{eq}}")
    st.markdown("**Tiempos**")
    st.latex(r"t_{\mathrm{par}}=\frac{\Delta n_m}{\dot n_m} \qquad t_{\mathrm{rampa}}=\frac{\Delta n_m}{\text{rampa}}")
    st.latex(r"t_{\mathrm{lim}}=\max\{t_{\mathrm{par}},\,t_{\mathrm{rampa}}\}")

with st.expander("Descripción — Sección 3"):
    st.markdown(
        "- **Aceleración por par**: incremento de rpm que brinda solo el par del motor.\n"
        "- **Tiempo por par**: si el par acelera libremente.\n"
        "- **Tiempo por rampa**: límite del VDF.\n"
        "- **Limitante**: el mayor de los dos."
    )

st.markdown("---")

# -----------------------------------------------------------------------------
# 4) Integración con carga hidráulica (slider en rpm, Q=k·n_p)
# -----------------------------------------------------------------------------
st.markdown("## 4) Integración con carga hidráulica")
st.latex(r"H(Q) = H_0 + K \left(\frac{Q}{3600}\right)^2 \qquad \big[\,Q:\mathrm{m^3/h},\ H:\mathrm{m}\,\big]")

# Modelo de eficiencia ---------------------------------------------------------
st.markdown("**Modelo de eficiencia**")
eta_mode = st.radio(
    "Seleccione el modelo para η",
    ["Polinómica (η_a, η_b, η_c)", "Constante (promedio del rango)", "Constante (fija por usuario)"],
    horizontal=True,
)
eta_user = None
if eta_mode == "Constante (fija por usuario)":
    eta_user = st.slider("η fija (usuario)", min_value=0.40, max_value=0.88, value=float(eta_beta if not np.isnan(eta_beta) else 0.72), step=0.01)

# Inputs fijos visuales
c4a, c4b, c4c, c4d = st.columns(4)
with c4a: st.markdown(f"- H₀: {val_blue(H0_m, 'm')}", unsafe_allow_html=True)
with c4b: st.markdown(f"- K: {val_blue(K_m_s2, 'm·s²')}", unsafe_allow_html=True)
with c4c: st.markdown(f"- ρ: {val_blue(rho, 'kg/m³', 0)}", unsafe_allow_html=True)
with c4d: st.markdown(f"- η (clip): {val_blue(eta_min*100, '%', 0)} – {val_blue(eta_max*100, '%', 0)}", unsafe_allow_html=True)

# Slider en rpm (25–50 Hz del TAG)
n_p_lo = float(n_p_min)
n_p_hi = float(n_p_max)
n_p_sel_lo, n_p_sel_hi = st.slider(
    "Rango de velocidad bomba n_p [rpm] (25–50 Hz)",
    min_value=n_p_lo, max_value=n_p_hi, value=(n_p_lo, n_p_hi), step=1.0
)

# Afinidad estricta Q = k·n_p (sin offset)
def affinity_k() -> float:
    if not np.isnan(Q_ref) and not np.isnan(n_ref_rpm) and n_ref_rpm > 0:
        return Q_ref / n_ref_rpm
    # ajuste por mínimos cuadrados con paso por el origen usando los puntos de 25–50 Hz
    n1, q1 = n_p_min, Q_min_ds
    n2, q2 = n_p_max, Q_max_ds
    nums = (n1 * q1 if (not np.isnan(n1) and not np.isnan(q1)) else 0.0) + \
           (n2 * q2 if (not np.isnan(n2) and not np.isnan(q2)) else 0.0)
    dens = (n1**2 if not np.isnan(n1) else 0.0) + (n2**2 if not np.isnan(n2) else 0.0)
    if dens <= 0:  # fallback
        return (Q_max_ds - Q_min_ds) / max(n_p_max - n_p_min, 1e-9)
    return nums / dens

k_aff = affinity_k()

# Mallas
N = 600
n_p_grid = np.linspace(n_p_sel_lo, n_p_sel_hi, N)
Q_grid = k_aff * n_p_grid                      # m3/h
q_grid = Q_grid / 3600.0
H_grid = H0_m + K_m_s2 * (q_grid ** 2)

def eta_from_Q(Q_m3h: np.ndarray) -> np.ndarray:
    # Polinómica si está elegida y hay coeficientes válidos
    if eta_mode.startswith("Polinómica") and not (
        np.isnan(eta_a) or np.isnan(eta_b) or np.isnan(eta_c) or np.isnan(Q_ref) or Q_ref <= 0
    ):
        beta = Q_m3h / Q_ref
        e = eta_a + eta_b * beta + eta_c * (beta ** 2)
    elif eta_mode.startswith("Constante (promedio"):
        # Promedio de la polinómica (si existe) o de eta_beta en el rango considerado
        if not (np.isnan(eta_a) or np.isnan(eta_b) or np.isnan(eta_c) or np.isnan(Q_ref) or Q_ref <= 0):
            beta = Q_m3h / Q_ref
            e_poly = eta_a + eta_b * beta + eta_c * (beta ** 2)
            e = np.full_like(Q_m3h, float(np.nanmean(e_poly)))
        elif not np.isnan(eta_beta):
            e = np.full_like(Q_m3h, float(eta_beta))
        else:
            e = np.full_like(Q_m3h, 0.72)
    else:
        # Constante fija por usuario
        e_fix = 0.72 if eta_user is None else float(eta_user)
        e = np.full_like(Q_m3h, e_fix)

    return np.clip(e, eta_min, eta_max)

eta_grid = eta_from_Q(Q_grid)
eta_min_used = float(np.min(eta_grid))
eta_max_used = float(np.max(eta_grid))
eta_mean_used = float(np.mean(eta_grid))

c_eta1, c_eta2, c_eta3 = st.columns(3)
with c_eta1: st.markdown(f"η mínima usada: {val_blue(eta_min_used*100, '%', 2)}", unsafe_allow_html=True)
with c_eta2: st.markdown(f"η máxima usada: {val_blue(eta_max_used*100, '%', 2)}", unsafe_allow_html=True)
with c_eta3: st.markdown(f"η media usada: {val_blue(eta_mean_used*100, '%', 2)}", unsafe_allow_html=True)

# Potencias y pares
P_h_grid = rho * g * q_grid * H_grid / np.maximum(eta_grid, 1e-6)   # W
omega_p  = 2.0 * math.pi * n_p_grid / 60.0
T_pump   = np.where(omega_p > 1e-9, P_h_grid / omega_p, 0.0)        # N·m (eje bomba)
T_load_m = T_pump / max(r_nm_np, 1e-9)                              # N·m (eje motor)
T_disp_m = np.full_like(T_load_m, T_nom_nm)                         # par motor constante

# Gráfico par vs n_p
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

# Integración temporal con chequeo de par suficiente
omega_m_grid = 2.0 * math.pi * (n_p_grid * r_nm_np) / 60.0  # ω_m = r*ω_p
T_net = T_disp_m - T_load_m
if np.any(T_net <= 0):
    idx_crit = np.argmax(T_net <= 0)
    ncrit = n_p_grid[idx_crit]
    st.warning(
        f"Par motor **insuficiente** en parte del rango seleccionado. "
        f"Primera intersección en n_p ≈ {fmt_num(ncrit, 'rpm', 0)}. "
        f"No se integra el tiempo."
    )
    t_hyd = float("nan")
else:
    d_omega_m = np.diff(omega_m_grid)
    dt = (J_eq * d_omega_m) / T_net[:-1]
    t_hyd = float(np.sum(dt))

# Resultados + bloque limitante debajo
c4_1, c4_2 = st.columns([1.2, 1.2])
with c4_1:
    st.markdown(f"- Tiempo por carga **hidráulica** (integración): {val_green(t_hyd, 's')}", unsafe_allow_html=True)
with c4_2:
    t_lim_4 = max(t_hyd, t_rampa) if not np.isnan(t_hyd) else t_rampa
    which = "hidráulica" if (not np.isnan(t_hyd) and t_hyd > t_rampa) else "rampa VDF"
    pill(f"Tiempo limitante (sección 4): <b>{which} = {fmt_num(t_lim_4, 's')}</b>")

# Gráfico dt/dn_m y área
if not np.any(T_net <= 0):
    n_m_grid = n_p_grid * r_nm_np
    dtdnm = (J_eq / T_net) * (2.0 * math.pi / 60.0)
    fig_area = go.Figure()
    fig_area.add_trace(go.Scatter(x=n_m_grid, y=dtdnm, mode="lines", name="dt/dn_m"))
    fig_area.add_trace(go.Scatter(x=n_m_grid, y=dtdnm, mode="lines", fill="tozeroy", name="Área integrada"))
    fig_area.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Velocidad motor n_m [rpm]",
        yaxis_title="dt/dn_m [s/rpm]",
        legend=dict(orientation="h"),
        height=320,
    )
    st.plotly_chart(fig_area, use_container_width=True)

with st.expander("Detalles y fórmulas — Sección 4"):
    st.markdown("**Curva del sistema**")
    st.latex(r"H(Q) = H_0 + K \left(\frac{Q}{3600}\right)^2 \quad\text{con}\quad K=\texttt{K\_m\_s2}")
    st.markdown("**Afinidad estricta (sin offset)**")
    st.latex(r"Q(n_p)=k\,n_p \qquad\text{con}\qquad k=\frac{Q_{\mathrm{ref}}}{n_{\mathrm{ref}}}\ \text{o ajuste por mínimos cuadrados}\;(\text{origen}).")
    st.markdown("**Eficiencia**")
    st.latex(r"\eta(Q) = \eta_a + \eta_b\,\beta + \eta_c\,\beta^2 \qquad \beta=\frac{Q}{Q_{\mathrm{ref}}}")
    st.markdown("Se acota a \( [\eta_{\min},\,\eta_{\max}] \).")
    st.markdown("**Potencia y par**")
    st.latex(r"P_h = \frac{\rho\,g\,Q_s\,H(Q)}{\eta(Q)}, \quad Q_s=\frac{Q}{3600}")
    st.latex(r"T_{\mathrm{pump}} = \frac{P_h}{\omega_p}, \ \omega_p=\frac{2\pi}{60}n_p, \ T_{\mathrm{load,m}}=\frac{T_{\mathrm{pump}}}{r}")
    st.markdown("**Par neto e integración**")
    st.latex(r"T_{\mathrm{net}} = T_{\mathrm{disp}} - T_{\mathrm{load,m}} \;\; (T_{\mathrm{net}}\le 0 \Rightarrow \text{no acelera})")
    st.latex(r"\frac{dt}{dn_m} = \frac{J_{eq}}{T_{\mathrm{net}}}\,\frac{2\pi}{60}, \qquad "
             r"t = \int_{n_{m,\min}}^{n_{m,\max}} \frac{dt}{dn_m}\,dn_m")

with st.expander("Descripción — Sección 4"):
    st.markdown(
        "- Con \(n_p\) en 25–50 Hz y **afinidad** \(Q=k\,n_p\) se genera \(H(Q)\) sin offsets.\n"
        "- Se calcula \(P_h\), \(T_{\mathrm{pump}}\) y el **par neto**; si siempre es positivo, integramos \(dt/dn_m\)."
    )

st.markdown("---")

# -----------------------------------------------------------------------------
# 5) Exportar CSV del TAG
# -----------------------------------------------------------------------------
st.markdown("## 5) Exportar datos del TAG")

res = {
    "TAG": tag,
    "pump_model": pump_model,
    "r_trans": r_nm_np,
    "motorpower_kw": P_motor_kW,
    "t_nom_nm": T_nom_nm,
    "n_m_min_rpm": n_m_min,
    "n_m_max_rpm": n_m_max,
    "n_p_min_rpm": n_p_min,
    "n_p_max_rpm": n_p_max,
    "H0_m": H0_m,
    "K_m_s2": K_m_s2,
    "rho_kgm3": rho,
    "eta_mode": eta_mode,
    "eta_user": eta_user if eta_user is not None else np.nan,
    "eta_min_used": eta_min_used,
    "eta_max_used": eta_max_used,
    "eta_mean_used": eta_mean_used,
    "Q_ref_m3h": Q_ref,
    "n_ref_rpm": n_ref_rpm,
    "k_aff_Q_over_rpm": k_aff,
    "n_p_sel_lo_rpm": n_p_sel_lo,
    "n_p_sel_hi_rpm": n_p_sel_hi,
    "J_m": J_m,
    "J_driver": J_dr,
    "J_bushing_driver": J_bdr,
    "J_driven": J_dn,
    "J_bushing_driven": J_bdn,
    "J_imp": J_imp,
    "J_eq": J_eq,
    "rampa_vdf_rpms": rampa_vdf,
    "t_por_par_s": t_par,
    "t_por_rampa_s": t_rampa,
    "t_hidraulica_s": t_hyd,
}

df_out = pd.DataFrame([res])
csv_bytes = df_out.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Descargar CSV del TAG", data=csv_bytes, file_name=f"{tag}_calculos.csv", mime="text/csv")
