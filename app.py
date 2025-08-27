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

# =============================================================================
# Utilidades
# =============================================================================

def dataset_path() -> Path:
    return Path(__file__).with_name("dataset.csv")

def images_path(name: str) -> Path:
    return Path(__file__).with_name("images") / name

def fmt_num(x, unit: str = "", ndigits: int = 2) -> str:
    """Formatea números con coma decimal; si es texto, lo devuelve tal cual."""
    if isinstance(x, str):
        return f"{x} {unit}".strip()
    if x is None:
        return "—"
    try:
        if isinstance(x, (int, float, np.floating)):
            if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
                return "—"
            s = f"{x:,.{ndigits}f}"
            s = s.replace(",", "_").replace(".", ",").replace("_", ".")
            return f"{s} {unit}".strip()
        return str(x)
    except Exception:
        return "—"

def color_value(text: str, color: str = BLUE, bold: bool = True) -> str:
    weight = "600" if bold else "400"
    return f'<span style="color:{color}; font-weight:{weight}">{text}</span>'

def val_blue(x, unit: str = "", ndigits: int = 2) -> str:
    return color_value(fmt_num(x, unit, ndigits), BLUE)

def val_green(x, unit: str = "", ndigits: int = 2) -> str:
    return color_value(fmt_num(x, unit, ndigits), GREEN)

# Caja de resultado/alerta
def pill(text: str, bg: str = "#e8f5e9", color: str = "#1b5e20"):
    st.markdown(
        f"""
        <div style="
            border-left: 5px solid {color};
            background:{bg};
            padding:0.8rem 1rem;
            border-radius:0.5rem;
            margin-top:0.5rem;">
            <span style="color:{color}; font-weight:600">{text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# Carga de datos
# =============================================================================
@st.cache_data
def load_data() -> pd.DataFrame:
    p = dataset_path()
    df = pd.read_csv(p, sep=";", decimal=",")
    # Normaliza nombres por si vienen con espacios/casos
    df.columns = [c.strip() for c in df.columns]
    return df

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

# Lectura de parámetros
pump_model = row.get("pumpmodel", "—")
r_trans    = float(row.get("r_trans", np.nan))
P_motor_kW = float(row.get("motorpower_kw", np.nan))
T_nom_nm   = float(row.get("t_nom_nm", np.nan))

n_m_min = float(row.get("motor_n_min_rpm", np.nan))
n_m_max = float(row.get("motor_n_max_rpm", np.nan))
n_p_min = float(row.get("pump_n_min_rpm", np.nan))
n_p_max = float(row.get("pump_n_max_rpm", np.nan))

imp_d_mm   = float(row.get("impeller_d_mm", np.nan))

# Inercias
J_m               = float(row.get("motor_j_kgm2", np.nan))
J_driver          = float(row.get("driverpulley_j_kgm2", np.nan))
J_bushing_driver  = float(row.get("driverbushing_j_kgm2", np.nan))
J_driven          = float(row.get("drivenpulley_j_kgm2", np.nan))
J_bushing_driven  = float(row.get("drivenbushing_j_kgm2", np.nan))
J_imp             = float(row.get("impeller_j_kgm2", np.nan))

# Hidráulica y eficiencia
H0_m      = float(row.get("H0_m", np.nan))
K_m_s2    = float(row.get("K_m_s2", np.nan))  # K del sistema (ver fórmula en sección 4)
Q_min_ds  = float(row.get("Q_min_m3h", np.nan))  # caudal a 25 Hz
Q_max_ds  = float(row.get("Q_max_m3h", np.nan))  # caudal a 50 Hz
Q_ref     = float(row.get("Q_ref_m3h", np.nan))

eta_a     = float(row.get("eta_a", np.nan))
eta_b     = float(row.get("eta_b", np.nan))
eta_c     = float(row.get("eta_c", np.nan))
eta_beta  = float(row.get("eta_beta", np.nan))
eta_min   = float(row.get("eta_min_clip", 0.40))
eta_max   = float(row.get("eta_max_clip", 0.88))

rho       = float(row.get("SlurryDensity_kgm3", np.nan))
if not (isinstance(rho, (int, float)) and rho > 0):
    rho = float(row.get("rho_kgm3", 1000.0))

g = 9.81

# Relación r robusta
if not (isinstance(r_trans, (int, float)) and r_trans > 0):
    # si faltara r, aproximamos por velocidades nominales
    if (isinstance(n_m_max, (int, float)) and isinstance(n_p_max, (int, float)) and n_p_max > 0):
        r_trans = n_m_max / n_p_max
    else:
        r_trans = 2.0

# =============================================================================
# 1) Parámetros
# =============================================================================

st.markdown("## 1) Parámetros")

c1, c2, c3 = st.columns([1.2, 1.2, 1.2])

with c1:
    st.markdown("**Identificación**")
    st.markdown(f"- Modelo de bomba: {val_blue(pump_model, '', 0)}", unsafe_allow_html=True)
    st.markdown(f"- TAG: {val_blue(tag_sel, '', 0)}", unsafe_allow_html=True)
    st.markdown(f"- Diámetro impulsor: {val_blue(imp_d_mm, 'mm', 0)}", unsafe_allow_html=True)

with c2:
    st.markdown("**Motor & transmisión**")
    st.markdown(f"- Potencia motor instalada: {val_blue(P_motor_kW, 'kW')}", unsafe_allow_html=True)
    st.markdown(f"- Par nominal del motor: {val_blue(T_nom_nm, 'N·m')}", unsafe_allow_html=True)
    st.markdown("- Relación transmisión:", unsafe_allow_html=True)
    st.latex(r"r \;=\; \dfrac{n_{\mathrm{motor}}}{n_{\mathrm{bomba}}}")
    st.markdown(f"&nbsp;&nbsp;{val_blue(r_trans, '')}", unsafe_allow_html=True)
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

    # Fallbacks 10% si faltan bushing
    if not (isinstance(J_bushing_driver, (int, float)) and J_bushing_driver > 0):
        J_bushing_driver = 0.10 * (J_driver if J_driver == J_driver else 0.0)
    if not (isinstance(J_bushing_driven, (int, float)) and J_bushing_driven > 0):
        J_bushing_driven = 0.10 * (J_driven if J_driven == J_driven else 0.0)

    st.markdown(f"- Motor (J_m): {val_blue(J_m, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Polea motriz (J_driver): {val_blue(J_driver, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- **Bushing** motriz (≈10% J_driver si no hay dato): {val_blue(J_bushing_driver, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Polea conducida (J_driven): {val_blue(J_driven, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- **Bushing** conducido (≈10% J_driven si no hay dato): {val_blue(J_bushing_driven, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Impulsor/rotor de bomba (J_imp): {val_blue(J_imp, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown("- Relación de velocidades:", unsafe_allow_html=True)
    st.latex(r"r \;=\; \dfrac{n_{\mathrm{motor}}}{n_{\mathrm{bomba}}}")
    st.markdown(f"&nbsp;&nbsp;{val_blue(r_trans, '')}", unsafe_allow_html=True)

    # J_eq (en eje motor): términos del lado bomba divididos por r^2
    J_eq = (J_m + J_driver + J_bushing_driver) + (J_driven + J_bushing_driven + J_imp) / (r_trans**2)
    st.markdown(f"**Inercia equivalente (J_eq):** {val_green(J_eq, 'kg·m²')}", unsafe_allow_html=True)

with colR:
    st.subheader("Fórmulas")
    st.latex(r"J_{\mathrm{eq}} \;=\; J_m \;+\; J_{\mathrm{driver}} \;+\; J_{\mathrm{bushing,driver}} \;+\; \dfrac{J_{\mathrm{driven}} + J_{\mathrm{bushing,driven}} + J_{\mathrm{imp}}}{r^2}")
    st.markdown("**Notas:**")
    st.latex(r"\text{Las inercias del lado bomba giran a }\ \omega_p=\omega_m/r.")
    st.latex(r"\text{Igualando energías cinéticas a una }\ \omega_m\ \text{común, los términos del lado de la bomba se dividen por } r^2.")
    with st.expander("Formulación de las inercias por componente", expanded=False):
        st.markdown(
            "- **Motor** (J_m): hojas de datos **WEG**.\n"
            "- **Poleas** (J_driver, J_driven): catálogo **TB Woods**.\n"
            "- **Bushing** (J_bushing_driver, J_bushing_driven): si no hay dato, se aproxima **10%** de la inercia de su polea.\n"
            "- **Impulsor** (J_imp): manuales **Metso**."
        )

# Descripción textual (Sección 2)
with st.expander("Descripción — Sección 2"):
    st.markdown(
        "La **inercia equivalente** \(J_{eq}\) es la suma de las inercias que ve el **eje del motor**. "
        "Las inercias del lado de la bomba se **reflejan** al eje del motor dividiéndolas por \(r^2\) debido a la relación de velocidades.\n\n"
        "Factores que aumentan \(J_{eq}\):\n"
        "- Inercias propias mayores (motor, poleas, bushings, impulsor).\n"
        "- **Relación** \(r\) más alta (más reducción hacia la bomba) reduce el impacto del lado bomba porque divide por \(r^2\).\n"
        "- Impulsores más grandes/pesados."
    )

st.markdown("---")

# =============================================================================
# 3) Tiempo inercial (par disponible vs rampa VDF)
# =============================================================================
st.markdown("## 3) Tiempo inercial (par disponible vs rampa VDF)")

rampa_vdf = st.slider("Rampa VDF en el motor [rpm/s]", min_value=10, max_value=600, value=100, step=5)

# Aceleración por par (rpm/s) y tiempos
if (isinstance(J_eq, (int, float)) and J_eq > 0) and (isinstance(T_nom_nm, (int, float)) and T_nom_nm > 0):
    n_dot_torque = (60.0 / (2.0 * math.pi)) * (T_nom_nm / J_eq)  # rpm/s
    t_par = (n_m_max - n_m_min) / max(n_dot_torque, 1e-12)       # s
else:
    n_dot_torque = float("nan")
    t_par = float("nan")

t_rampa = (n_m_max - n_m_min) / max(float(rampa_vdf), 1e-12)

cA, cB, cC = st.columns(3)
with cA:
    st.markdown(f"- Aceleración por par (lado motor): {val_green(n_dot_torque, 'rpm/s')}", unsafe_allow_html=True)
with cB:
    st.markdown(f"- Tiempo por par (25→50 Hz): {val_green(t_par, 's')}", unsafe_allow_html=True)
with cC:
    st.markdown(f"- Tiempo por rampa VDF: {val_blue(t_rampa, 's')}", unsafe_allow_html=True)

# Detalles y fórmulas — Sección 3
with st.expander("Detalles y fórmulas — Sección 3"):
    st.markdown("**Hipótesis**: par del motor constante en 25–50 Hz, pérdidas mecánicas despreciables.")
    st.latex(r"J_{eq}\,\dot\omega_m \;=\; T_{\mathrm{disp}} \;\Rightarrow\; \dot\omega_m \;=\; \dfrac{T_{\mathrm{disp}}}{J_{eq}}")
    st.latex(r"n_m \;=\; \dfrac{60}{2\pi}\,\omega_m \;\Rightarrow\; \dot n_m \;=\; \dfrac{60}{2\pi}\,\dfrac{T_{\mathrm{disp}}}{J_{eq}}")
    st.latex(r"t_{\mathrm{par}} \;=\; \dfrac{\Delta n_m}{\dot n_m} \qquad;\qquad t_{\mathrm{rampa}} \;=\; \dfrac{\Delta n_m}{\text{rampa}}")
    st.latex(r"\text{Criterio: } t = \max\{\,t_{\mathrm{par}},\ t_{\mathrm{rampa}}\,\}")

# Resultado: tiempo limitante (Sección 3)
lim3 = "Tiempo limitante (sección 3): "
if not np.isnan(t_par) and (t_par > t_rampa):
    pill(lim3 + f"<b>por par</b> = {fmt_num(t_par, 's')}")
else:
    pill(lim3 + f"<b>por rampa VDF</b> = {fmt_num(t_rampa, 's')}")

# Descripción textual (Sección 3)
with st.expander("Descripción — Sección 3"):
    st.markdown(
        "La **aceleración por par** es la razón de cambio de velocidad producida por el par disponible del motor sobre la inercia equivalente. "
        "El **tiempo por par** es el tiempo requerido para ir de 25 a 50 Hz usando únicamente ese par. "
        "El **tiempo por rampa** es el delimitado por el VDF; el criterio de diseño suele ser el **mayor** entre ambos."
    )

st.markdown("---")

# =============================================================================
# 4) Integración con carga hidráulica
# =============================================================================
st.markdown("## 4) Integración con carga hidráulica")

# Fórmulas iniciales
st.latex(r"H(Q) \;=\; H_0 \;+\; K\,\left(\dfrac{Q}{3600}\right)^2 \qquad \big[\,Q:\ \mathrm{m^3/h},\ H:\ \mathrm{m}\,\big]")
st.latex(r"\eta(Q) \approx \eta_a + \eta_b\,\beta + \eta_c\,\beta^2 \quad\text{con}\quad \beta=\dfrac{Q}{Q_{\mathrm{ref}}},\ \eta\in[\eta_{\min},\eta_{\max}]")
st.latex(r"P_h \;=\; \dfrac{\rho\,g\,Q_s\,H(Q)}{\eta(Q)} \quad;\quad Q_s=\dfrac{Q}{3600}")
st.latex(r"T_{\mathrm{pump}} \;=\; \dfrac{P_h}{\omega_p},\ \ \omega_p=\dfrac{2\pi n_p}{60} \quad\Rightarrow\quad T_{\mathrm{load,m}}=\dfrac{T_{\mathrm{pump}}}{r}")
st.latex(r"T_{\mathrm{net}} \;=\; T_{\mathrm{disp}}-T_{\mathrm{load,m}} \quad;\quad \Delta t \;=\; \dfrac{J_{eq}\,\Delta\omega_m}{T_{\mathrm{net}}}")
st.latex(r"\frac{dt}{dn_m} \;=\; \dfrac{J_{eq}}{T_{\mathrm{net}}}\,\dfrac{2\pi}{60} \quad\Rightarrow\quad t=\int_{n_{m,\min}}^{n_{m,\max}} \frac{dt}{dn_m}\,dn_m")

# Parámetros mostrados
c4a, c4b, c4c, c4d = st.columns(4)
with c4a:
    st.markdown(f"- H₀: {val_blue(H0_m, 'm')}", unsafe_allow_html=True)
with c4b:
    st.markdown(f"- K (K_m_s2): {val_blue(K_m_s2, 'm·s²')}", unsafe_allow_html=True)
with c4c:
    st.markdown(f"- ρ: {val_blue(rho, 'kg/m³', 0)}", unsafe_allow_html=True)
with c4d:
    st.markdown(f"- η clip: {val_blue(eta_min*100, '%', 0)} – {val_blue(eta_max*100, '%', 0)}", unsafe_allow_html=True)

# Slicer de caudal limitado a 25–50 Hz del TAG
# (Q_min_m3h y Q_max_m3h vienen del dataset para 25 Hz y 50 Hz)
qmin_slider = float(Q_min_ds) if np.isfinite(Q_min_ds) else 0.0
qmax_slider = float(Q_max_ds) if np.isfinite(Q_max_ds) else max(1000.0, qmin_slider + 100.0)
if qmax_slider <= qmin_slider:
    qmax_slider = qmin_slider + 1.0

Q_min, Q_max = st.slider(
    "Rango de caudal considerado [m³/h] (limitado a 25–50 Hz del TAG)",
    min_value=float(qmin_slider),
    max_value=float(qmax_slider),
    value=(float(qmin_slider), float(qmax_slider)),
    step=1.0,
)

# Funciones auxiliares
def eta_from_Q(Q_m3h: np.ndarray) -> np.ndarray:
    # Polinomio si hay coeficientes; si no, valor constante (eta_beta) o 0.72
    if np.isfinite(eta_a) and np.isfinite(eta_b) and np.isfinite(eta_c) and np.isfinite(Q_ref) and Q_ref > 0:
        beta = Q_m3h / Q_ref
        e = eta_a + eta_b * beta + eta_c * (beta**2)
    elif np.isfinite(eta_beta) and eta_beta > 0:
        e = np.full_like(Q_m3h, float(eta_beta))
    else:
        e = np.full_like(Q_m3h, 0.72)
    # Clipping
    emin = eta_min if np.isfinite(eta_min) else 0.40
    emax = eta_max if np.isfinite(eta_max) else 0.88
    return np.clip(e, emin, emax)

# Mallas
N = 600
n_m_grid = np.linspace(n_m_min, n_m_max, N)
n_p_grid = n_m_grid / max(r_trans, 1e-12)

# Mapeo lineal Q(n_p) dentro del rango elegido en el slider (ligado a 25–50 Hz)
def Q_from_np(n_p: np.ndarray) -> np.ndarray:
    if n_p_max <= n_p_min + 1e-9:
        return np.full_like(n_p, Q_min)
    # Interpolamos Q entre Q_min y Q_max cuando n_p va de n_p_min a n_p_max
    return Q_min + (Q_max - Q_min) * (n_p - n_p_min) / (n_p_max - n_p_min)

Q_grid   = Q_from_np(n_p_grid)
q_grid_s = Q_grid / 3600.0
H_grid   = H0_m + K_m_s2 * (q_grid_s**2)
eta_grid = eta_from_Q(Q_grid)

# Potencias y pares
P_h_grid = rho * g * q_grid_s * H_grid / np.maximum(eta_grid, 1e-9)   # W
omega_p  = 2.0 * math.pi * n_p_grid / 60.0                            # rad/s
T_pump   = np.where(omega_p > 1e-12, P_h_grid / omega_p, np.inf)      # N·m (eje bomba)
T_load_m = T_pump / max(r_trans, 1e-12)                               # N·m (eje motor)
T_disp_m = np.full_like(T_load_m, T_nom_nm)                           # par motor (constante en 25–50 Hz)

# Gráfico: Par en el eje motor vs n_p
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

# Gráfico: Potencia hidráulica
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
omega_m_grid = 2.0 * math.pi * n_m_grid / 60.0
d_omega      = np.diff(omega_m_grid)
T_net        = T_disp_m - T_load_m

# Si en algún punto T_net <= 0 ⇒ par insuficiente: no hay aceleración
if np.any(T_net <= 0):
    idx_first = int(np.argmax(T_net <= 0))
    n_m_block = n_m_grid[idx_first]
    st.markdown(f"- Tiempo por carga **hidráulica** (integración): {val_green(np.nan, 's')}", unsafe_allow_html=True)
    pill(f"Par motor **insuficiente** a partir de \(n_m \approx {fmt_num(n_m_block, 'rpm', 0)}\). No es posible completar la aceleración en 25–50 Hz.", bg="#fdecea", color="#b71c1c")
    t_hyd = float("nan")
else:
    dt = (J_eq * d_omega) / np.maximum(T_net[:-1], 1e-9)
    t_hyd = float(np.sum(dt))
    st.markdown(f"- Tiempo por carga **hidráulica** (integración): {val_green(t_hyd, 's')}", unsafe_allow_html=True)
    # Bloque verde inmediatamente debajo
    t_lim_4 = max(t_hyd, t_rampa) if not np.isnan(t_hyd) else t_rampa
    which = "hidráulica" if (not np.isnan(t_hyd) and t_hyd > t_rampa) else "rampa VDF"
    pill(f"Tiempo limitante (sección 4): <b>{which}</b> = {fmt_num(t_lim_4, 's')}", bg="#e8f5e9", color="#1b5e20")

# Curva integranda: dt/dn_m = J_eq*(2π/60)/T_net
dt_dn = (J_eq * (2.0 * math.pi / 60.0)) / np.maximum(T_net, 1e-9)
fig_int = go.Figure()
fig_int.add_trace(go.Scatter(x=n_m_grid, y=dt_dn, mode="lines", name="dt/dn_m [s/rpm]", fill="tozeroy"))
fig_int.update_layout(
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis_title="Velocidad motor n_m [rpm]",
    yaxis_title="dt/dn_m [s/rpm]",
    legend=dict(orientation="h"),
    height=320,
)
st.plotly_chart(fig_int, use_container_width=True)

# Detalles — Sección 4
with st.expander("Detalles y fórmulas — Sección 4"):
    st.markdown("**Curva del sistema** y **eficiencia**:")
    st.latex(r"H(Q)=H_0+K\left(\dfrac{Q}{3600}\right)^2,\quad K=K_{\mathrm{m\_s^2}}")
    st.latex(r"Q\propto n_p\ \text{(25–50 Hz)}\ \Rightarrow\ \text{interpolación lineal entre } n_{p,\min}\ \text{y}\ n_{p,\max}.")
    st.latex(r"\eta(Q)=\eta_a+\eta_b\beta+\eta_c\beta^2,\ \beta=\dfrac{Q}{Q_{\mathrm{ref}}},\ \eta\in[\eta_{\min},\eta_{\max}]")
    st.markdown("**Potencias, pares y dinámica:**")
    st.latex(r"P_h=\dfrac{\rho g Q_s H(Q)}{\eta(Q)},\quad Q_s=\dfrac{Q}{3600}")
    st.latex(r"T_{\mathrm{pump}}=\dfrac{P_h}{\omega_p},\ \ \omega_p=\dfrac{2\pi n_p}{60},\ \ T_{\mathrm{load,m}}=\dfrac{T_{\mathrm{pump}}}{r}")
    st.latex(r"T_{\mathrm{net}}=T_{\mathrm{disp}}-T_{\mathrm{load,m}};\quad \dot\omega_m=\dfrac{T_{\mathrm{net}}}{J_{eq}};\quad \Delta t=\dfrac{J_{eq}\Delta\omega_m}{T_{\mathrm{net}}}")
    st.latex(r"\frac{dt}{dn_m}=\dfrac{J_{eq}}{T_{\mathrm{net}}}\dfrac{2\pi}{60}\ \Rightarrow\ t=\int_{n_{m,\min}}^{n_{m,\max}}\frac{dt}{dn_m}\,dn_m")

# Descripción textual (Sección 4)
with st.expander("Descripción — Sección 4"):
    st.markdown(
        "En 25–50 Hz se asume **\(T_{disp}\)** constante. Para cada \(n_p\) asociado, estimamos **\(Q\)** por afinidad, "
        "luego evaluamos la **carga del sistema** \(H(Q)\) y la **eficiencia** \(\\eta(Q)\). Con esto obtenemos "
        "la **potencia hidráulica** y el **par de bomba**, que se reflejan al eje del motor. La diferencia con el par disponible "
        "es el **par neto**; a partir de él integramos \(dt/dn_m\) para obtener el **tiempo de aceleración**. "
        "Si \(T_{net}\le0\) en algún punto, el par del motor es **insuficiente** para completar el barrido 25–50 Hz."
    )

st.markdown("---")

# =============================================================================
# 5) Exportar CSV con datos calculados (TAG actual)
# =============================================================================
st.markdown("## 5) Exportar resultados del TAG")

# Compilamos una fila de resultados clave
results = {
    "TAG": tag_sel,
    "pump_model": pump_model,
    "r_trans": r_trans,
    "motorpower_kw": P_motor_kW,
    "t_nom_nm": T_nom_nm,
    "n_m_min_rpm": n_m_min,
    "n_m_max_rpm": n_m_max,
    "n_p_min_rpm": n_p_min,
    "n_p_max_rpm": n_p_max,
    "J_m": J_m,
    "J_driver": J_driver,
    "J_bushing_driver": J_bushing_driver,
    "J_driven": J_driven,
    "J_bushing_driven": J_bushing_driven,
    "J_imp": J_imp,
    "J_eq": J_eq,
    "H0_m": H0_m,
    "K_m_s2": K_m_s2,
    "Q_min_m3h_slider": Q_min,
    "Q_max_m3h_slider": Q_max,
    "Q_min_m3h_TAG(25Hz)": Q_min_ds,
    "Q_max_m3h_TAG(50Hz)": Q_max_ds,
    "Q_ref_m3h": Q_ref,
    "eta_a": eta_a,
    "eta_b": eta_b,
    "eta_c": eta_c,
    "eta_beta": eta_beta,
    "eta_min_clip": eta_min,
    "eta_max_clip": eta_max,
    "rho_kgm3": rho,
    "rampa_vdf_rpmps": rampa_vdf,
    "n_dot_torque_rpmps": n_dot_torque,
    "t_par_s": t_par,
    "t_rampa_s": t_rampa,
    "t_hid_s": t_hyd,
}

df_out = pd.DataFrame([results])
csv_bytes = df_out.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Descargar CSV del TAG",
    data=csv_bytes,
    file_name=f"reporte_{tag_sel}.csv",
    mime="text/csv",
)
