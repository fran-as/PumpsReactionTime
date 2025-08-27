# app.py
# ─────────────────────────────────────────────────────────────────────────────
# Dashboard: Tiempo de reacción de bombas con VDF (modelo P(Q))
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
BLUE  = "#1f77b4"  # valores dados → azul
GREEN = "#2ca02c"  # valores calculados → verde
GRAY  = "#6c757d"

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
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    s = f"{x:,.{ndigits}f}".replace(",", "_").replace(".", ",").replace("_", ".")
    return f"{s} {unit}".strip()

def color_value(text: str, color: str = BLUE, bold: bool = True) -> str:
    w = "600" if bold else "400"
    return f'<span style="color:{color}; font-weight:{w}">{text}</span>'

def val_blue(x, unit="", ndigits=2) -> str:
    return color_value(fmt_num(x, unit, ndigits), BLUE)

def val_green(x, unit="", ndigits=2) -> str:
    return color_value(fmt_num(x, unit, ndigits), GREEN)

def pill(text: str, bg: str = "#e8f5e9", color: str = "#1b5e20"):
    # No usar Markdown (**) dentro: solo HTML para evitar ** visibles
    st.markdown(
        f"""
        <div style="border-left: 5px solid {color}; background:{bg}; padding:0.8rem 1rem; border-radius:0.5rem; margin-top:0.5rem">
            <span style="color:{color}; font-weight:600">{text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

@st.cache_data
def load_data() -> pd.DataFrame:
    p = dataset_path()
    df = pd.read_csv(p, sep=";", decimal=",")
    # Validación de columnas mínimas
    needed = [
        "TAG","pumpmodel","impeller_d_mm","motorpower_kw","t_nom_nm","r_trans",
        "motor_n_min_rpm","motor_n_max_rpm","pump_n_min_rpm","pump_n_max_rpm",
        "motor_j_kgm2","driverpulley_j_kgm2","driverbushing_j_kgm2",
        "drivenpulley_j_kgm2","drivenbushing_j_kgm2","impeller_j_kgm2",
        "Q_min_m3h","Q_max_m3h",
        "P_a0_kW","P_a1_kW_per_m3h","P_a2_kW_per_m3h2","P_a3_kW_per_m3h3",
        "P_order","P_R2",
        "SlurryDensity_kgm3",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.error(f"Faltan columnas en dataset.csv: {missing}")
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
pump_model    = row["pumpmodel"]
impeller_d_mm = float(row["impeller_d_mm"])
motor_kw      = float(row["motorpower_kw"])
t_nom         = float(row["t_nom_nm"])
r_nm_np       = float(row["r_trans"])

n_m_min = float(row["motor_n_min_rpm"])
n_m_max = float(row["motor_n_max_rpm"])
n_p_min = float(row["pump_n_min_rpm"])
n_p_max = float(row["pump_n_max_rpm"])

J_m      = float(row["motor_j_kgm2"])
J_driver = float(row["driverpulley_j_kgm2"])
J_bdrv   = float(row["driverbushing_j_kgm2"])
J_driven = float(row["drivenpulley_j_kgm2"])
J_bdrn   = float(row["drivenbushing_j_kgm2"])
J_imp    = float(row["impeller_j_kgm2"])

Q_min    = float(row["Q_min_m3h"])
Q_max    = float(row["Q_max_m3h"])

# Coeficientes de la potencia al eje P(Q) en kW
a0 = float(row["P_a0_kW"])
a1 = float(row["P_a1_kW_per_m3h"])
a2 = float(row["P_a2_kW_per_m3h2"])
a3 = float(row["P_a3_kW_per_m3h3"])
P_order = int(row["P_order"])
P_R2    = float(row["P_R2"])

rho_slurry = float(row["SlurryDensity_kgm3"])

# =============================================================================
# 1) Parámetros
# =============================================================================

st.markdown("## 1) Parámetros")

c1, c2, c3 = st.columns([1.2, 1.2, 1.2])

with c1:
    st.markdown("**Identificación**")
    st.markdown(f"- Modelo de bomba: {val_blue(pump_model, '', 0)}", unsafe_allow_html=True)
    st.markdown(f"- TAG: {val_blue(tag_sel, '', 0)}", unsafe_allow_html=True)
    st.markdown(f"- Diámetro impulsor: {val_blue(impeller_d_mm, 'mm')}", unsafe_allow_html=True)

with c2:
    st.markdown("**Motor & transmisión**")
    st.markdown(f"- Potencia motor instalada: {val_blue(motor_kw, 'kW')}", unsafe_allow_html=True)
    st.markdown(f"- Par nominal del motor: {val_blue(t_nom, 'N·m')}", unsafe_allow_html=True)
    st.markdown(f"- Relación transmisión: {val_blue(r_nm_np, '')}", unsafe_allow_html=True)
    st.latex(r"r \;=\; \dfrac{n_m}{n_p}")
    st.markdown(f"- Velocidad motor min–max: {val_blue(n_m_min, 'rpm', 0)} – {val_blue(n_m_max, 'rpm', 0)}", unsafe_allow_html=True)

with c3:
    st.markdown("**Bomba (25–50 Hz por afinidad)**")
    st.markdown(f"- Velocidad bomba min–max: {val_blue(n_p_min, 'rpm', 0)} – {val_blue(n_p_max, 'rpm', 0)}", unsafe_allow_html=True)
    st.markdown(f"- Caudal min–max: {val_blue(Q_min, 'm³/h', 0)} – {val_blue(Q_max, 'm³/h', 0)}", unsafe_allow_html=True)
    st.markdown(f"- Densidad de pulpa ρ: {val_blue(rho_slurry, 'kg/m³', 0)}", unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# 2) Cálculo de inercia equivalente
# =============================================================================
st.header("2) Cálculo de inercia equivalente")

colL, colR = st.columns([1.1, 1])

with colL:
    st.subheader("Inercias individuales")

    # Fallbacks de bushing (si vinieran NaN): 10% de su polea
    if np.isnan(J_bdrv): J_bdrv = 0.10 * J_driver
    if np.isnan(J_bdrn): J_bdrn = 0.10 * J_driven

    st.markdown(f"- Motor (J_m): {val_blue(J_m, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Polea motriz (J_driver): {val_blue(J_driver, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Bushing motriz (≈10% J_driver): {val_blue(J_bdrv, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Polea conducida (J_driven): {val_blue(J_driven, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Bushing conducido (≈10% J_driven): {val_blue(J_bdrn, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Impulsor/rotor (J_imp): {val_blue(J_imp, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Relación r: {val_blue(r_nm_np, '')}", unsafe_allow_html=True)
    st.latex(r"r \;=\; \dfrac{n_m}{n_p}")

    # J_eq en el eje del motor (inercias del lado bomba vistas /r²)
    J_eq = (J_m + J_driver + J_bdrv) + (J_driven + J_bdrn + J_imp) / (r_nm_np**2)
    st.markdown(f"**Inercia equivalente (J_eq):** {val_green(J_eq, 'kg·m²')}", unsafe_allow_html=True)

with colR:
    st.subheader("Fórmula utilizada")
    st.latex(r"J_{\mathrm{eq}} \;=\; J_m \;+\; J_{\mathrm{driver}} \;+\; J_{\mathrm{bushing,driver}} \;+\; \dfrac{J_{\mathrm{driven}} + J_{\mathrm{bushing,driven}} + J_{\mathrm{imp}}}{r^2}")
    with st.expander("Formulación de las inercias por componente", expanded=True):
        st.markdown(
            "- **Motor**: hojas de datos **WEG**.\n"
            "- **Poleas**: catálogo **TB Woods**.\n"
            "- **Bushing**: se aproxima **10%** de la polea asociada.\n"
            "- **Impulsor**: manuales **Metso**.\n\n"
            "Las inercias del lado bomba giran a \(\\omega_p=\\omega_m/r\\). Igualando energías cinéticas a una \(\\omega_m\\) común resulta la división por \(r^2\) para las del lado de la bomba."
        )

st.markdown("---")

# =============================================================================
# 3) Tiempo inercial (par disponible vs rampa VDF)
# =============================================================================

st.markdown("## 3) Tiempo inercial (par disponible vs rampa VDF)")
rampa_vdf = st.slider("Rampa VDF en el motor [rpm/s]", min_value=10, max_value=600, value=300, step=5)

n_dot_torque = (60.0 / (2.0 * math.pi)) * (t_nom / J_eq)   # rpm/s (par constante en 25–50 Hz)
t_par   = (n_m_max - n_m_min) / max(n_dot_torque, 1e-9)
t_rampa = (n_m_max - n_m_min) / max(float(rampa_vdf), 1e-9)

cA, cB, cC = st.columns(3)
with cA:
    st.markdown(f"- Aceleración por par (lado motor): {val_green(n_dot_torque, 'rpm/s')}", unsafe_allow_html=True)
with cB:
    st.markdown(f"- Tiempo por par (25→50 Hz): {val_green(t_par, 's')}", unsafe_allow_html=True)
with cC:
    st.markdown(f"- Tiempo por rampa VDF: {val_blue(t_rampa, 's')}", unsafe_allow_html=True)

with st.expander("Descripción — Sección 3"):
    st.markdown("Hipótesis: par del motor constante (Tdisp=Tnom) en 25–50 Hz y pérdidas mecánicas despreciables.")
    st.latex(r"J_{\rm eq}\,\dot\omega_m=T_{\rm disp}\Rightarrow \dot\omega_m=\tfrac{T_{\rm disp}}{J_{\rm eq}}")
    st.latex(r"n_m=\tfrac{60}{2\pi}\,\omega_m\Rightarrow \dot n_m=\tfrac{60}{2\pi}\,\tfrac{T_{\rm disp}}{J_{\rm eq}}")
    st.latex(r"t_{\rm par}=\dfrac{\Delta n_m}{\dot n_m}\qquad t_{\rm rampa}=\dfrac{\Delta n_m}{\text{rampa}}")

# Bloque verde (Sección 3)
if t_par > t_rampa:
    pill(f"Tiempo limitante (sección 3): por par = {fmt_num(t_par, 's')}")
else:
    pill(f"Tiempo limitante (sección 3): por rampa VDF = {fmt_num(t_rampa, 's')}")

st.markdown("---")

# =============================================================================
# 4) Carga hidráulica con modelo P(Q)
# =============================================================================

st.markdown("## 4) Carga hidráulica con modelo de potencia \(P(Q)\)")

# Slider en velocidad bomba (25–50 Hz del TAG)
n_p_lo, n_p_hi = st.slider(
    "Rango de velocidad de bomba [rpm] (limitado a 25–50 Hz del TAG)",
    min_value=float(n_p_min), max_value=float(n_p_max),
    value=(float(n_p_min), float(n_p_max)), step=1.0,
)

# Mallas y afinidad Q(n_p)
N = 600
n_p_grid = np.linspace(n_p_lo, n_p_hi, N)
Q_grid   = Q_min + (Q_max - Q_min) * (n_p_grid - n_p_min) / max((n_p_max - n_p_min), 1e-9)

# Potencia al eje P(Q) [kW] respetando el grado del ajuste
a0 = float(row["P_a0_kW"]); a1 = float(row["P_a1_kW_per_m3h"])
a2 = float(row["P_a2_kW_per_m3h2"]); a3 = float(row["P_a3_kW_per_m3h3"])
order = int(row["P_order"])
if order == 1:
    P_kW = a0 + a1*Q_grid
elif order == 2:
    P_kW = a0 + a1*Q_grid + a2*(Q_grid**2)
else:
    P_kW = a0 + a1*Q_grid + a2*(Q_grid**2) + a3*(Q_grid**3)
P_kW = np.maximum(P_kW, 0.0)

# Par de bomba y reflejado al motor
omega_p  = 2.0 * math.pi * n_p_grid / 60.0
T_pump   = np.where(omega_p > 1e-9, (P_kW * 1000.0) / omega_p, 0.0)
T_load_m = T_pump / max(r_nm_np, 1e-9)
T_motor  = np.full_like(T_load_m, t_nom)

# Gráfico: Par vs n_p
fig_t = go.Figure()
fig_t.add_trace(go.Scatter(x=n_p_grid, y=T_load_m, name="Par resistente reflejado (motor)", mode="lines"))
fig_t.add_trace(go.Scatter(x=n_p_grid, y=T_motor,   name="Par motor disponible", mode="lines"))
fig_t.update_layout(
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis_title="Velocidad bomba n_p [rpm]",
    yaxis_title="Par en el eje motor [N·m]",
    legend=dict(orientation="h", y=1.02, yanchor="bottom"),
    height=360,
)
st.plotly_chart(fig_t, use_container_width=True)

# Gráfico: Potencia vs n_p
fig_p = go.Figure()
fig_p.add_trace(go.Scatter(x=n_p_grid, y=P_kW, mode="lines", name="P(Q) [kW]"))
fig_p.update_layout(
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis_title="Velocidad bomba n_p [rpm]",
    yaxis_title="Potencia al eje [kW]",
    legend=dict(orientation="h"),
    height=320,
)
st.plotly_chart(fig_p, use_container_width=True)

# Integración temporal (siempre que T_net>0 en todo el rango)
n_m_grid   = n_p_grid * r_nm_np
omega_m    = 2.0 * math.pi * n_m_grid / 60.0
T_net      = T_motor - T_load_m

if np.any(T_net <= 0.0):
    st.markdown(f"- Tiempo por carga **hidráulica** (integración): {val_green(float('nan'), 's')}", unsafe_allow_html=True)
    st.markdown(f"- Tiempo por **rampa VDF** (para comparación): {val_blue(t_rampa, 's')}", unsafe_allow_html=True)

    with st.expander("Descripción — Sección 4", expanded=False):
        st.markdown(
            "Se usa la **potencia al eje** como función del caudal ajustada a puntos del TAG. "
            "El caudal se obtiene por afinidad estricta en 25–50 Hz."
        )
        st.latex(r"P(Q)=a_0 + a_1\,Q + a_2\,Q^2 + a_3\,Q^3\quad (\text{kW})")
        st.latex(r"Q(n_p) = Q_{\min} + (Q_{\max}-Q_{\min})\,\dfrac{n_p-n_{p,\min}}{n_{p,\max}-n_{p,\min}}")
        st.latex(r"T_{\rm pump}(n_p)=\dfrac{P(Q(n_p))\cdot 1000}{\omega_p},\ \ \omega_p=\dfrac{2\pi n_p}{60},\ \ T_{\rm load,m}=\dfrac{T_{\rm pump}}{r}")
        st.latex(r"\dot\omega_m=\dfrac{T_{\rm disp}-T_{\rm load,m}}{J_{\rm eq}},\ \ \Delta t=J_{\rm eq}\,\dfrac{\Delta\omega_m}{T_{\rm net}},\ \ \frac{dt}{dn_m}=J_{\rm eq}\,\frac{2\pi}{60}\,\frac{1}{T_{\rm net}}")
        st.markdown("Si \(T_{net}\le 0\) en el rango seleccionado, el motor no puede acelerar: **par insuficiente**.")
    pill("Tiempo limitante (sección 4): par insuficiente en el rango seleccionado. Ajuste el rango o verifique el TAG.")
else:
    d_omega = np.diff(omega_m)
    dt = (J_eq * d_omega) / T_net[:-1]
    t_hyd = float(np.sum(dt))

    # Mostrar tiempos comparables
    st.markdown(f"- Tiempo por carga **hidráulica** (integración): {val_green(t_hyd, 's')}", unsafe_allow_html=True)
    st.markdown(f"- Tiempo por **rampa VDF** (para comparación): {val_blue(t_rampa, 's')}", unsafe_allow_html=True)

    # Curva dt/dn_m y área integrada
    dtdn = (J_eq * (2.0 * math.pi / 60.0)) / T_net
    fig_dt = go.Figure()
    fig_dt.add_trace(go.Scatter(x=n_m_grid, y=dtdn, mode="lines", name="dt/dn_m [s/rpm]"))
    fig_dt.add_trace(go.Scatter(
        x=np.concatenate([n_m_grid, n_m_grid[::-1]]),
        y=np.concatenate([dtdn, np.zeros_like(dtdn)[::-1]]),
        fill="toself", name="área integrada (t)", opacity=0.2, showlegend=True
    ))
    fig_dt.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Velocidad motor n_m [rpm]",
        yaxis_title="dt/dn_m [s/rpm]",
        legend=dict(orientation="h"),
        height=320,
    )
    st.plotly_chart(fig_dt, use_container_width=True)

    # Descripción y luego bloque verde (mismo patrón que Sección 3)
    with st.expander("Descripción — Sección 4", expanded=False):
        st.markdown(
            "Se usa la **potencia al eje** como función del caudal a partir de puntos de operación del TAG; "
            "se prescinde de la eficiencia explícita y se trabaja con \(P(Q)\) en kW."
        )
        st.latex(r"P(Q)=a_0 + a_1\,Q + a_2\,Q^2 + a_3\,Q^3")
        st.latex(r"Q(n_p) = Q_{\min} + (Q_{\max}-Q_{\min})\,\dfrac{n_p-n_{p,\min}}{n_{p,\max}-n_{p,\min}}")
        st.latex(r"T_{\rm pump}(n_p)=\dfrac{P(Q(n_p))\cdot 1000}{\omega_p},\quad \omega_p=\dfrac{2\pi n_p}{60},\qquad T_{\rm load,m}=\dfrac{T_{\rm pump}}{r}")
        st.latex(r"\dot\omega_m=\dfrac{T_{\rm disp}-T_{\rm load,m}}{J_{\rm eq}},\qquad \Delta t=J_{\rm eq}\,\dfrac{\Delta\omega_m}{T_{\rm net}},\qquad \frac{dt}{dn_m}=J_{\rm eq}\,\frac{2\pi}{60}\,\frac{1}{T_{\rm net}}")

    # Bloque verde con el tiempo limitante (hidráulica vs rampa)
    if t_hyd > t_rampa:
        pill(f"Tiempo limitante (sección 4): hidráulica = {fmt_num(t_hyd, 's')}")
    else:
        pill(f"Tiempo limitante (sección 4): rampa VDF = {fmt_num(t_rampa, 's')}")

st.markdown("---")
