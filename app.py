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

# Paleta
BLUE  = "#1f77b4"   # Dado (dataset) → azul
GREEN = "#2ca02c"   # Calculado      → verde
GRAY  = "#6c757d"

# Utilidades ------------------------------------------------------------------

def color_value(text: str, color: str = BLUE, bold: bool = True) -> str:
    w = "600" if bold else "400"
    return f'<span style="color:{color}; font-weight:{w}">{text}</span>'

def fmt_num(x, unit: str = "", ndigits: int = 2) -> str:
    """Formatea números con coma decimal; si x es texto, lo devuelve tal cual."""
    # Si es texto, no lo intentes formatear como número
    if isinstance(x, str):
        return f"{x} {unit}".strip()
    # None o NaN/Inf → raya
    if x is None:
        return "—"
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(xf) or math.isinf(xf):
        return "—"
    # Formato numérico con coma decimal
    s = f"{xf:,.{ndigits}f}"
    s = s.replace(",", "_").replace(".", ",").replace("_", ".")
    return f"{s} {unit}".strip()

def val_blue(x, unit="", ndigits=2) -> str:
    return color_value(fmt_num(x, unit, ndigits), BLUE)

def val_green(x, unit="", ndigits=2) -> str:
    return color_value(fmt_num(x, unit, ndigits), GREEN)


# =============================================================================
# Mapeo de columnas (exactos del dataset)
# =============================================================================
COL = {
    "TAG": "TAG",
    "pump_model": "pumpmodel",
    "impeller_d_mm": "impeller_d_mm",
    "motorpower_kw": "motorpower_kw",
    "t_nom_nm": "t_nom_nm",
    "r": "r_trans",
    "n_m_min": "motor_n_min_rpm",
    "n_m_max": "motor_n_max_rpm",
    "n_p_min": "pump_n_min_rpm",
    "n_p_max": "pump_n_max_rpm",
    "J_m": "motor_j_kgm2",
    "J_driver": "driverpulley_j_kgm2",
    "J_bushing_driver": "driverbushing_j_kgm2",
    "J_driven": "drivenpulley_j_kgm2",
    "J_bushing_driven": "drivenbushing_j_kgm2",
    "J_imp": "impeller_j_kgm2",
    "H0_m": "H0_m",
    "K_m_s2": "K_m_s2",
    "R2_H": "R2_H",
    "eta_a": "eta_a",
    "eta_b": "eta_b",
    "eta_c": "eta_c",
    "R2_eta": "R2_eta",
    "Q_min_m3h": "Q_min_m3h",
    "Q_max_m3h": "Q_max_m3h",
    "Q_ref_m3h": "Q_ref_m3h",
    "n_ref_rpm": "n_ref_rpm",
    "rho": "rho_kgm3",
    "eta_beta": "eta_beta",
    "eta_min_clip": "eta_min_clip",
    "eta_max_clip": "eta_max_clip",
    "SlurryDensity": "SlurryDensity_kgm3",
}

NUM_KEYS = {
    "impeller_d_mm","motorpower_kw","t_nom_nm","r",
    "n_m_min","n_m_max","n_p_min","n_p_max",
    "J_m","J_driver","J_bushing_driver","J_driven","J_bushing_driven","J_imp",
    "H0_m","K_m_s2","R2_H","eta_a","eta_b","eta_c","R2_eta",
    "Q_min_m3h","Q_max_m3h","Q_ref_m3h","n_ref_rpm","rho",
    "eta_beta","eta_min_clip","eta_max_clip","SlurryDensity",
}

def load_data() -> pd.DataFrame:
    # dataset.csv con sep=';' y decimal=','
    df = pd.read_csv(dataset_path(), sep=";", decimal=",")
    return df

def get_val(row: pd.Series, key: str, default=np.nan):
    col = COL[key]
    v = row.get(col, default)
    try:
        return float(v) if key in NUM_KEYS else ("" if pd.isna(v) else str(v))
    except Exception:
        return default

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
tags = df[COL["TAG"]].astype(str).tolist()
tag_sel = st.sidebar.selectbox("Selecciona TAG", tags, index=0)
row = df[df[COL["TAG"]].astype(str) == str(tag_sel)].iloc[0]

# Valores base (datos/derivados)
pump_model   = get_val(row, "pump_model", "")
D_imp_mm     = get_val(row, "impeller_d_mm")
P_motor_kW   = get_val(row, "motorpower_kw")
T_nom_nm     = get_val(row, "t_nom_nm")
r_nm_np      = get_val(row, "r", np.nan)

n_m_min = get_val(row, "n_m_min")
n_m_max = get_val(row, "n_m_max")
n_p_min = get_val(row, "n_p_min")
n_p_max = get_val(row, "n_p_max")

# Inercias (con fallback 10% para bushing si faltan)
J_m    = get_val(row, "J_m", 0.0)
J_drv  = get_val(row, "J_driver", 0.0)
J_bdrv = get_val(row, "J_bushing_driver", np.nan)
if np.isnan(J_bdrv): J_bdrv = 0.10 * J_drv
J_dvn  = get_val(row, "J_driven", 0.0)
J_bdvn = get_val(row, "J_bushing_driven", np.nan)
if np.isnan(J_bdvn): J_bdvn = 0.10 * J_dvn
J_imp  = get_val(row, "J_imp", 0.0)

# Hidráulica + eficiencia
H0_m   = get_val(row, "H0_m")
K_m_s2 = get_val(row, "K_m_s2")
Qmin_d = get_val(row, "Q_min_m3h")
Qmax_d = get_val(row, "Q_max_m3h")
Q_ref  = get_val(row, "Q_ref_m3h")
n_ref  = get_val(row, "n_ref_rpm")
rho    = get_val(row, "rho", 1000.0)
eta_a  = get_val(row, "eta_a", np.nan)
eta_b  = get_val(row, "eta_b", np.nan)
eta_c  = get_val(row, "eta_c", np.nan)
eta_bta= get_val(row, "eta_beta", np.nan)
eta_min= get_val(row, "eta_min_clip", 0.40)
eta_max= get_val(row, "eta_max_clip", 0.88)

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
    st.markdown(f"- Diámetro impulsor: {val_blue(D_imp_mm, 'mm')}", unsafe_allow_html=True)

with c2:
    st.markdown("**Motor & transmisión**")
    st.markdown(f"- Potencia motor instalada: {val_blue(P_motor_kW, 'kW')}", unsafe_allow_html=True)
    st.markdown(f"- Par nominal del motor: {val_blue(T_nom_nm, 'N·m')}", unsafe_allow_html=True)
    st.markdown(
        f"- Relación transmisión \( r = n_\\mathrm{{motor}}/n_\\mathrm{{bomba}} \): {val_blue(r_nm_np, '')}",
        unsafe_allow_html=True,
    )
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

st.caption(
    "En el rango 25–50 Hz el par del motor puede asumirse aproximadamente **constante**, "
    "lo que simplifica la comparación entre **par disponible** y **par resistente**."
)
st.markdown("---")

# =============================================================================
# 2) Cálculo de inercia equivalente
# =============================================================================
st.header("2) Cálculo de inercia equivalente")

colL, colR = st.columns([1.1, 1])

with colL:
    st.subheader("Inercias individuales")
    st.markdown(f"- Motor (J_m): {val_blue(J_m, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Polea motriz (J_driver): {val_blue(J_drv, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Bushing motriz (≈10% de J_driver): {val_blue(J_bdrv, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Polea conducida (J_driven): {val_blue(J_dvn, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Bushing conducido (≈10% de J_driven): {val_blue(J_bdvn, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Impulsor/rotor de bomba (J_imp): {val_blue(J_imp, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Relación \( r=n_m/n_p \): {val_blue(r_nm_np, '')}", unsafe_allow_html=True)

    # J_eq en el eje del motor (inercias del lado bomba vistas /r²)
    r_safe = max(r_nm_np if not np.isnan(r_nm_np) else 1.0, 1e-9)
    J_eq = (J_m + J_drv + J_bdrv) + (J_dvn + J_bdvn + J_imp) / (r_safe**2)
    st.markdown(f"**Inercia equivalente (J_eq):** {val_green(J_eq, 'kg·m²')}", unsafe_allow_html=True)

with colR:
    st.subheader("Fórmula utilizada")
    st.latex(r"J_{\mathrm{eq}} = J_m + J_{\mathrm{driver}} + J_{\mathrm{bushing,driver}} + \dfrac{J_{\mathrm{driven}}+J_{\mathrm{bushing,driven}}+J_{\mathrm{imp}}}{r^2}")
    st.markdown(
        "Las inercias del lado bomba giran a \(\\omega_p = \\omega_m/r\\). "
        "Igualando energías cinéticas a una \(\\omega_m\\) común, los términos del lado de la bomba se dividen por \(r^2\)."
    )

st.markdown("---")

# =============================================================================
# 3) Tiempo inercial (par disponible vs rampa VDF)
# =============================================================================
st.markdown("## 3) Tiempo inercial (par disponible vs rampa VDF)")

rampa_vdf = st.slider("Rampa VDF en el motor [rpm/s]", min_value=10, max_value=600, value=100, step=5)

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

lim3 = "Tiempo limitante (sección 3): "
if not np.isnan(t_par) and t_par > t_rampa:
    pill(lim3 + f"<b>por par</b> = {fmt_num(t_par, 's')}")
else:
    pill(lim3 + f"<b>por rampa VDF</b> = {fmt_num(t_rampa, 's')}")

with st.expander("Detalles y fórmulas — Sección 3", expanded=False):
    st.markdown(
        r"""
- **Hipótesis:** par del motor constante \(T_{\mathrm{disp}}=T_{\mathrm{nom}}\) en 25–50 Hz.
- **Dinámica rotacional (eje motor):** \(J_{eq}\,\dot\omega_m=T_{\mathrm{disp}}\Rightarrow \dot\omega_m=\tfrac{T_{\mathrm{disp}}}{J_{eq}}\).
- **Conversión a rpm:** \(n_m=\tfrac{60}{2\pi}\,\omega_m \Rightarrow \dot n_m=\tfrac{60}{2\pi}\,\tfrac{T_{\mathrm{disp}}}{J_{eq}}\).
- **Tiempo por par:** \(t_{\mathrm{par}}=\dfrac{\Delta n_m}{\dot n_m}\).
- **Tiempo por rampa VDF:** \(t_{\mathrm{rampa}}=\dfrac{\Delta n_m}{\text{rampa}}\).
- **Criterio:** se toma el **mayor** entre \(t_{\mathrm{par}}\) y \(t_{\mathrm{rampa}}\).
        """
    )

st.markdown("---")

# =============================================================================
# 4) Integración con carga hidráulica (slider limitado a 25–50 Hz)
# =============================================================================
st.markdown("## 4) Integración con carga hidráulica")

# Fórmulas principales
st.latex(r"H(Q) = H_0 + K\left(\dfrac{Q}{3600}\right)^2 \quad [\,Q:\ \mathrm{m^3/h},\ H:\ \mathrm{m}\,]")
st.latex(r"P_h(Q) = \dfrac{\rho g\,Q_s\,H(Q)}{\eta(Q)},\quad Q_s=\dfrac{Q}{3600}")
st.latex(r"T_{\mathrm{pump}}(Q)=\dfrac{P_h(Q)}{\omega_p},\quad \omega_p=\dfrac{2\pi n_p}{60},\quad T_{\mathrm{load,m}}=\dfrac{T_{\mathrm{pump}}}{r}")

# Fila de parámetros
c4a, c4b, c4c, c4d = st.columns(4)
with c4a:
    st.markdown(f"- H₀: {val_blue(H0_m, 'm')}", unsafe_allow_html=True)
with c4b:
    st.markdown(f"- K: {val_blue(K_m_s2, 'm·s²')}", unsafe_allow_html=True)
with c4c:
    st.markdown(f"- ρ: {val_blue(rho, 'kg/m³', 0)}", unsafe_allow_html=True)
with c4d:
    st.markdown(f"- η (clip): {val_blue(eta_min*100, '%', 0)} – {val_blue(eta_max*100, '%', 0)}", unsafe_allow_html=True)

# ----- Slider de caudal limitado a los caudales a 25 y 50 Hz (afinidad)
# Usamos Q_ref @ n_ref y escalamos a n_p_min / n_p_max
def flows_from_affinity(Q_ref_m3h, n_ref_rpm, n_p_min_rpm, n_p_max_rpm):
    if any(np.isnan(x) for x in [Q_ref_m3h, n_ref_rpm, n_p_min_rpm, n_p_max_rpm]) or n_ref_rpm <= 0:
        # Fallback a dataset Q_min, Q_max si existen
        return float(Qmin_d), float(Qmax_d)
    q_min = Q_ref_m3h * (n_p_min_rpm / n_ref_rpm)
    q_max = Q_ref_m3h * (n_p_max_rpm / n_ref_rpm)
    # Asegurar orden
    lo, hi = (q_min, q_max) if q_min <= q_max else (q_max, q_min)
    return float(lo), float(hi)

Q_min_25, Q_max_50 = flows_from_affinity(Q_ref, n_ref, n_p_min, n_p_max)

# Slider restringido a ese intervalo; por defecto, extremos completos
Q_min_sel, Q_max_sel = st.slider(
    "Rango de caudal considerado [m³/h] (limitado a 25–50 Hz)",
    min_value=float(max(0.0, Q_min_25)),
    max_value=float(max(Q_min_25 + 1.0, Q_max_50)),
    value=(float(Q_min_25), float(Q_max_50)),
    step=1.0,
)

# Mapear Q con n_p por afinidad dentro del rango 25–50 Hz
def Q_from_np(n_p: np.ndarray, n_p_min, n_p_max, Q_lo, Q_hi) -> np.ndarray:
    if n_p_max <= n_p_min + 1e-9:
        return np.full_like(n_p, Q_lo)
    return Q_lo + (Q_hi - Q_lo) * (n_p - n_p_min) / (n_p_max - n_p_min)

def eta_from_Q(Q_m3h: np.ndarray) -> np.ndarray:
    if not (np.isnan(eta_a) or np.isnan(eta_b) or np.isnan(eta_c) or np.isnan(Q_ref) or Q_ref <= 0):
        beta = Q_m3h / Q_ref
        e = eta_a + eta_b * beta + eta_c * (beta**2)
    elif not np.isnan(eta_bta) and eta_bta > 0:
        e = np.full_like(Q_m3h, float(eta_bta))
    else:
        e = np.full_like(Q_m3h, 0.72)
    return np.clip(e, eta_min, eta_max)

# Mallas
N = 600
n_m_grid = np.linspace(n_m_min, n_m_max, N)
r_safe = max(r_nm_np if not np.isnan(r_nm_np) else 1.0, 1e-9)
n_p_grid = n_m_grid / r_safe

Q_grid   = Q_from_np(n_p_grid, n_p_min, n_p_max, Q_min_sel, Q_max_sel)
q_grid_s = Q_grid / 3600.0
H_grid   = H0_m + K_m_s2 * (q_grid_s**2)
eta_grid = eta_from_Q(Q_grid)

# Potencias y pares
P_h_grid = rho * g * q_grid_s * H_grid / np.maximum(eta_grid, 1e-6)  # W
omega_p  = 2.0 * math.pi * n_p_grid / 60.0                           # rad/s
T_pump   = np.where(omega_p > 1e-9, P_h_grid / omega_p, 0.0)         # N·m (bomba)
T_load_m = T_pump / r_safe                                           # N·m (motor)
T_disp_m = np.full_like(T_load_m, T_nom_nm)

# Gráfico de pares
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

# Potencia hidráulica
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

# Integración temporal con verificación de suficiencia de par
col_time, = st.columns(1)
with col_time:
    # Si en algún punto T_net <= 0, no integramos (par insuficiente)
    T_net = T_disp_m - T_load_m
    if np.any(T_net <= 0):
        idx = np.where(T_net <= 0)[0]
        n_crit = n_p_grid[idx[0]] if len(idx) else np.nan
        st.warning(
            f"Par motor **insuficiente** en parte del rango (ej.: n_p≈{fmt_num(n_crit,'rpm',0)}). "
            "No se calcula el tiempo de aceleración con carga hidráulica para este rango de caudales."
        )
        t_hyd = float("nan")
    else:
        # Integración: Δt = J_eq Δω_m / T_net
        omega_m = 2.0 * math.pi * n_m_grid / 60.0
        d_omega = np.diff(omega_m)
        dt = (J_eq * d_omega) / T_net[:-1]
        t_hyd = float(np.sum(dt))

        st.markdown(f"- Tiempo por carga **hidráulica** (integración): {val_green(t_hyd, 's')}",
                    unsafe_allow_html=True)

        # Curva dt/dn_m y área integrada
        dn = np.diff(n_m_grid)
        dtdn = (J_eq * (2.0*math.pi/60.0)) / T_net  # s / rpm

        fig_area = go.Figure()
        # Línea dt/dn_m
        fig_area.add_trace(go.Scatter(
            x=n_m_grid, y=dtdn, mode="lines", name="dt/dn_m [s/rpm]"
        ))
        # Área integrada (trapecios)
        fig_area.add_trace(go.Scatter(
            x=np.concatenate([n_m_grid, n_m_grid[::-1]]),
            y=np.concatenate([dtdn, np.zeros_like(dtdn)[::-1]]),
            fill="toself",
            mode="lines",
            name="Área integrada (≈ tiempo)",
            opacity=0.25
        ))
        fig_area.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Velocidad motor n_m [rpm]",
            yaxis_title="dt/dn_m [s/rpm]",
            legend=dict(orientation="h"),
            height=320,
        )
        st.plotly_chart(fig_area, use_container_width=True)

        # Bloque de “tiempo limitante” justo DEBAJO del tiempo hidráulico:
        t_lim_4 = max(t_hyd, t_rampa) if not np.isnan(t_hyd) else t_rampa
        which = "hidráulica" if (not np.isnan(t_hyd) and t_hyd > t_rampa) else "rampa VDF"
        pill(f"Tiempo limitante (sección 4): <b>{which} = {fmt_num(t_lim_4, 's')}</b>")

with st.expander("Detalles y fórmulas — Sección 4", expanded=False):
    st.markdown(
        r"""
- **Curva del sistema:** \(H(Q)=H_0+K\,(Q/3600)^2\), con \(K=K_{\mathrm{m\,s^2}}\) del dataset.
- **Afinidad (25–50 Hz):** \(Q\propto n_p\). En el slider, los extremos se fijan con
  \(Q_{25} = Q_{\rm ref}\,n_{p,\min}/n_{\rm ref}\) y \(Q_{50} = Q_{\rm ref}\,n_{p,\max}/n_{\rm ref}\).
- **Eficiencia:** si hay coeficientes, se usa \( \eta(Q)=\eta_a+\eta_b\beta+\eta_c\beta^2 \)
  con \( \beta=Q/Q_{\rm ref}\); en caso contrario se usa el valor de dataset (acotado a \([0{,}40,\,0{,}88]\)).
- **Potencia hidráulica:** \(P_h=\rho g\,Q_s\,H(Q)/\eta(Q)\), \(Q_s=Q/3600\).
- **Par de bomba y reflejo al motor:** \( T_{\rm pump}=P_h/\omega_p, \ \omega_p=2\pi n_p/60,\ T_{\rm load,m}=T_{\rm pump}/r\).
- **Par neto:** \( T_{\rm net}=T_{\rm disp}-T_{\rm load,m} \). Si \( T_{\rm net}\le 0 \) en algún punto, no hay aceleración.
- **Integración temporal:** \( \Delta t = J_{eq}\,\Delta\omega_m/T_{\rm net} \).
  El integrando es \( \dfrac{dt}{dn_m} = \dfrac{J_{eq}(2\pi/60)}{T_{\rm net}} \);
  el área entre \(n_{m,\min}\) y \(n_{m,\max}\) da el tiempo total.
        """
    )

st.markdown("---")

# =============================================================================
# 5) Exportar CSV del TAG
# =============================================================================
st.markdown("## 5) Exportar resultados del TAG")
res = {
    "TAG": tag_sel,
    "pump_model": pump_model,
    "r_nm_np": r_nm_np,
    "n_m_min_rpm": n_m_min, "n_m_max_rpm": n_m_max,
    "n_p_min_rpm": n_p_min, "n_p_max_rpm": n_p_max,
    "J_m": J_m, "J_driver": J_drv, "J_bushing_driver": J_bdrv,
    "J_driven": J_dvn, "J_bushing_driven": J_bdvn, "J_imp": J_imp,
    "J_eq": J_eq,
    "P_motor_kW": P_motor_kW, "T_nom_nm": T_nom_nm,
    "H0_m": H0_m, "K_m_s2": K_m_s2, "rho_kgm3": rho,
    "eta_a": eta_a, "eta_b": eta_b, "eta_c": eta_c,
    "eta_min_clip": eta_min, "eta_max_clip": eta_max,
    "Q_ref_m3h": Q_ref, "n_ref_rpm": n_ref,
    "Q_min_25_m3h": Q_min_25, "Q_max_50_m3h": Q_max_50,
    "Q_slider_min_m3h": Q_min_sel, "Q_slider_max_m3h": Q_max_sel,
    "n_dot_torque_rpms": n_dot_torque,
    "t_par_s": t_par, "t_rampa_s": t_rampa,
}
# Si se calculó t_hyd:
if 't_hyd' in locals() and not np.isnan(t_hyd):
    res["t_hidraulica_s"] = t_hyd

df_out = pd.DataFrame([res])
csv_bytes = df_out.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Descargar CSV del TAG",
    csv_bytes,
    file_name=f"reporte_{tag_sel}.csv",
    mime="text/csv",
)

# =============================================================================
# Notas
# =============================================================================
st.caption(
    "Notas: el slider de la Sección 4 queda **acotado automáticamente** a los "
    "caudales equivalentes a **25 Hz** y **50 Hz** via afinidad. Dentro de ese rango "
    "el par del motor se considera **constante**, lo que simplifica y robustece el análisis."
)
