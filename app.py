# app.py
# ─────────────────────────────────────────────────────────────────────────────
# Memoria de Cálculo – Tiempo de reacción (VDF) para bombas
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# Configuración general
# =============================================================================
st.set_page_config(page_title="Memoria de Cálculo – Tiempo de reacción (VDF)", layout="wide")

# Paleta
BLUE = "#1f77b4"   # Dado (dataset)
GREEN = "#2ca02c"  # Calculado
GRAY = "#6c757d"

# =============================================================================
# Utilidades
# =============================================================================
BASE_DIR = Path(__file__).parent

def dataset_path() -> Path:
    return BASE_DIR / "dataset.csv"

def images_path(name: str) -> Path:
    return BASE_DIR / "images" / name

def get_num(x) -> float:
    """Convierte cadenas (con decimal ',') a float. Devuelve NaN si no aplica."""
    if x is None:
        return float("nan")
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace(" ", "").replace("\u00a0", "")
    s = s.replace(",", ".")
    s = re.sub(r"[^0-9eE\+\-\.]", "", s)
    try:
        return float(s)
    except Exception:
        return float("nan")

def fmt_num(x, unit: str = "", ndigits: int = 2) -> str:
    """Formatea con coma decimal; si es texto, lo devuelve tal cual."""
    if isinstance(x, str):
        return f"{x} {unit}".strip()
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    s = f"{x:,.{ndigits}f}".replace(",", "_").replace(".", ",").replace("_", ".")
    return f"{s} {unit}".strip()

def color_value(text: str, color: str = BLUE, bold: bool = True) -> str:
    w = "600" if bold else "400"
    return f"<span style='color:{color}; font-weight:{w}'>{text}</span>"

def val_blue(x, unit: str = "", ndigits: int = 2) -> str:
    return color_value(fmt_num(x, unit, ndigits), BLUE)

def val_green(x, unit: str = "", ndigits: int = 2) -> str:
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

# =============================================================================
# Carga de dataset (csv con sep=';' y decimal=',')
# =============================================================================
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    path = dataset_path()
    df = pd.read_csv(path, sep=";", decimal=",", dtype=str)
    # Normalizamos nombres por si llegan con may/min mezclado
    df.columns = [c.strip() for c in df.columns]
    return df

# =============================================================================
# Helpers específicos de Sección 4
# =============================================================================
def k_flow_ratio(row: pd.Series) -> float:
    """
    Devuelve k tal que Q(n_p) = k * n_p (m3/h por rpm).
    Se calcula promediando (Q_min/n_p_min, Q_max/n_p_max). Si faltan,
    cae a (Q_ref/n_ref). Devuelve 0.0 si no se puede.
    """
    def g(col, default=np.nan):
        return get_num(row.get(col, default))

    qmin = g("Q_min_m3h")
    qmax = g("Q_max_m3h")
    npmin = g("pump_n_min_rpm")
    npmax = g("pump_n_max_rpm")
    k_vals = []
    if not (np.isnan(qmin) or np.isnan(npmin) or npmin <= 0):
        k_vals.append(qmin / npmin)
    if not (np.isnan(qmax) or np.isnan(npmax) or npmax <= 0):
        k_vals.append(qmax / npmax)
    if k_vals:
        return float(np.mean(k_vals))

    qref = g("Q_ref_m3h")
    nref = g("n_ref_rpm")
    if not (np.isnan(qref) or np.isnan(nref) or nref <= 0):
        return qref / nref
    return 0.0

def eta_eval(Q_m3h: np.ndarray, row: pd.Series, mode: str, eta_user: float | None) -> np.ndarray:
    """
    Calcula eficiencia según 'mode':
      - 'poly'     : polinomio del dataset (clipped)
      - 'poly-avg' : promedio del polinomio en el tramo (constante)
      - 'fixed'    : constante provista (o 'eta' del dataset si existe)
    """
    a = get_num(row.get("eta_a", np.nan))
    b = get_num(row.get("eta_b", np.nan))
    c = get_num(row.get("eta_c", np.nan))
    qref = get_num(row.get("Q_ref_m3h", np.nan))
    eta_clip_min = get_num(row.get("eta_min_clip", 0.40))
    eta_clip_max = get_num(row.get("eta_max_clip", 0.88))
    base_eta = get_num(row.get("eta", np.nan))  # por si alguna versión trae 'eta'

    if mode == "fixed":
        eta_val = eta_user if (eta_user and eta_user > 0) else (base_eta if not np.isnan(base_eta) else 0.72)
        return np.full_like(Q_m3h, float(np.clip(eta_val, eta_clip_min, eta_clip_max)))

    # Polinomio
    if not np.isnan(a) and not np.isnan(b) and not np.isnan(c) and not np.isnan(qref) and qref > 0:
        beta = Q_m3h / qref
        eta_poly = a + b * beta + c * (beta ** 2)
        eta_poly = np.clip(eta_poly, eta_clip_min, eta_clip_max)
    else:
        eta_poly = np.full_like(Q_m3h, base_eta if not np.isnan(base_eta) else 0.72)

    if mode == "poly-avg":
        const = float(np.mean(eta_poly))
        return np.full_like(Q_m3h, const)

    return eta_poly  # 'poly'

# =============================================================================
# Encabezado (logos + título)
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

# Accesos rápidos a valores de fila
def R(col, default=np.nan) -> float:
    return get_num(row.get(col, default))

def S(col, default="—") -> str:
    v = row.get(col, default)
    return "—" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)

g = 9.81

# =============================================================================
# 1) Parámetros
# =============================================================================
st.markdown("## 1) Parámetros")
c1, c2, c3 = st.columns([1.2, 1.2, 1.2])

pump_model = S("pumpmodel")
impeller_d_mm = R("impeller_d_mm")
P_motor_kW = R("motorpower_kw")
T_nom_nm = R("t_nom_nm")
r_nm_np = R("r_trans")

n_m_min = R("motor_n_min_rpm")
n_m_max = R("motor_n_max_rpm")
n_p_min = R("pump_n_min_rpm")
n_p_max = R("pump_n_max_rpm")

rho = R("SlurryDensity_kgm3")
if np.isnan(rho) or rho <= 0:
    rho = R("rho_kgm3")
if np.isnan(rho) or rho <= 0:
    rho = 1000.0

with c1:
    st.markdown("**Identificación**")
    st.markdown(f"- Modelo de bomba: {val_blue(pump_model, '', 0)}", unsafe_allow_html=True)
    st.markdown(f"- TAG: {val_blue(tag_sel, '', 0)}", unsafe_allow_html=True)
    st.markdown(f"- Diámetro impulsor: {val_blue(impeller_d_mm, 'mm')}", unsafe_allow_html=True)

with c2:
    st.markdown("**Motor & transmisión**")
    st.markdown(f"- Potencia motor instalada: {val_blue(P_motor_kW, 'kW')}", unsafe_allow_html=True)
    st.markdown(f"- Par nominal del motor: {val_blue(T_nom_nm, 'N·m')}", unsafe_allow_html=True)
    st.markdown(
        rf"- Relación transmisión \(r=\frac{{n_{{motor}}}}{{n_{{bomba}}}}\): {val_blue(r_nm_np, '')}",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"- Velocidad motor min–max: {val_blue(n_m_min, 'rpm', 0)} – {val_blue(n_m_max, 'rpm', 0)}",
        unsafe_allow_html=True,
    )

with c3:
    st.markdown("**Bomba (25–50 Hz)**")
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

colL2, colR2 = st.columns([1.1, 1])

with colL2:
    st.subheader("Inercias individuales")

    J_m = R("motor_j_kgm2")
    J_driver = R("driverpulley_j_kgm2")
    J_bushing_driver = R("driverbushing_j_kgm2")
    if np.isnan(J_bushing_driver):  # aproximación 10% del catálogo
        J_bushing_driver = 0.10 * J_driver

    J_driven = R("drivenpulley_j_kgm2")
    J_bushing_driven = R("drivenbushing_j_kgm2")
    if np.isnan(J_bushing_driven):
        J_bushing_driven = 0.10 * J_driven

    J_imp = R("impeller_j_kgm2")
    r = max(r_nm_np, 1e-6)

    st.markdown(f"- Motor (J_m): {val_blue(J_m, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Polea motriz (J_driver): {val_blue(J_driver, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Bushing motriz (≈10% J_driver): {val_blue(J_bushing_driver, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Polea conducida (J_driven): {val_blue(J_driven, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Bushing conducido (≈10% J_driven): {val_blue(J_bushing_driven, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(f"- Impulsor/rotor de bomba (J_imp): {val_blue(J_imp, 'kg·m²')}", unsafe_allow_html=True)
    st.markdown(rf"- Relación \(r=\frac{{n_m}}{{n_p}}\): {val_blue(r, '')}", unsafe_allow_html=True)

    # Inercia equivalente en eje motor (lado bomba dividido por r^2)
    J_eq = (J_m + J_driver + J_bushing_driver) + (J_driven + J_bushing_driven + J_imp) / (r ** 2)
    st.markdown(f"**Inercia equivalente (J_eq):** {val_green(J_eq, 'kg·m²')}", unsafe_allow_html=True)

with colR2:
    st.subheader("Fórmula utilizada")
    st.latex(
        r"J_{\mathrm{eq}} = J_m + J_{\mathrm{driver}} + J_{\mathrm{bushing,driver}}"
        r"+ \frac{J_{\mathrm{driven}} + J_{\mathrm{bushing,driven}} + J_{\mathrm{imp}}}{r^2}"
    )
    with st.expander("Formulación de las inercias por componente", expanded=True):
        st.markdown(
            "- **Poleas**: inercias desde catálogo **TB Woods**.\n"
            "- **Bushing**: si falta dato, se aproxima al **10%** de la inercia de su polea.\n"
            "- **Impulsor**: de manuales **Metso**.\n\n"
            r"Las inercias del lado bomba giran a \(\omega_p=\omega_m/r\). "
            r"Igualando energías cinéticas a una \(\omega_m\) común, los términos del lado de la bomba se dividen por \(r^2\)."
        )

# =============================================================================
# 3) Tiempo inercial (par disponible vs rampa VDF)
# =============================================================================
st.markdown("## 3) Tiempo inercial (par disponible vs rampa VDF)")
rampa_vdf = st.slider("Rampa VDF en el motor [rpm/s]", min_value=10, max_value=600, value=100, step=5)

# Aceleración por par y tiempos
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
    pill(lim3 + f"{color_value('por par', GREEN)} = {fmt_num(t_par, 's')}")
else:
    pill(lim3 + f"{color_value('por rampa VDF', GREEN)} = {fmt_num(t_rampa, 's')}")

with st.expander("Detalles y fórmulas — Sección 3"):
    st.markdown(
        r"""
**Hipótesis.** Par del motor constante \(T_{\mathrm{disp}}=T_{\mathrm{nom}}\) en 25–50 Hz; pérdidas mecánicas despreciables.

**Dinámica rotacional (eje motor).** \(J_{eq}\,\dot\omega_m = T_{\mathrm{disp}}\Rightarrow \dot\omega_m = T_{\mathrm{disp}}/J_{eq}\).

**Conversión a rpm.** Con \(n_m=\frac{60}{2\pi}\omega_m\) resulta \(\dot n_m = \frac{60}{2\pi}\frac{T_{\mathrm{disp}}}{J_{eq}}\).

**Tiempo por par.** \(t_{\mathrm{par}}=\dfrac{\Delta n_m}{\dot n_m}\).  
**Tiempo por rampa VDF.** \(t_{\mathrm{rampa}}=\dfrac{\Delta n_m}{\text{rampa}}\).  
**Criterio.** Tiempo limitante = \(\max\{t_{\mathrm{par}},\,t_{\mathrm{rampa}}\}\).
        """
    )

st.markdown("---")

# =============================================================================
# 4) Integración con carga hidráulica (afinidad estricta Q = k·n)
# =============================================================================
st.markdown("## 4) Integración con carga hidráulica")
st.latex(r"H(Q)=H_0+K\left(\frac{Q}{3600}\right)^2,\quad Q\,[\mathrm{m^3/h}],\ H\,[\mathrm{m}]")
st.latex(r"Q(n_p)=k\,n_p,\ \ \omega_p=\tfrac{2\pi}{60}\,n_p,\ \ P_h=\frac{\rho g\,Q/3600 \cdot H(Q)}{\eta(Q)},\ \ T_{\text{pump}}=\frac{P_h}{\omega_p},\ \ T_{\text{load,m}}=\tfrac{T_{\text{pump}}}{r}")

H0_m = R("H0_m")
K_m_s2 = R("K_m_s2")

# Afinidad estricta: Q = k * n_p
k = k_flow_ratio(row)
if k <= 0:
    st.warning("No se pudo determinar el factor de afinidad k (Q/n). Verifica Q_min/Q_max y n_p_min/n_p_max en el dataset.")
    k = 1.0  # evita crash, aunque no sea físico

Q_25Hz = k * n_p_min
Q_50Hz = k * n_p_max

# Subtramo seleccionado (en rpm)
n_sel_min, n_sel_max = st.slider(
    "Tramo de operación (25–50 Hz) en **rpm** de bomba (se muestra Q equivalente)",
    min_value=float(n_p_min), max_value=float(n_p_max),
    value=(float(n_p_min), float(n_p_max)), step=1.0,
)
st.caption(
    f"Equivalente en caudal: {fmt_num(k*n_sel_min, 'm³/h', 0)} → {fmt_num(k*n_sel_max, 'm³/h', 0)} "
    f"(límite físico del TAG: {fmt_num(Q_25Hz, 'm³/h', 0)} → {fmt_num(Q_50Hz, 'm³/h', 0)})"
)

# Modelo de eficiencia
eta_mode = st.radio(
    "Modelo de eficiencia η(Q):", ["Polinomio (dataset)", "Constante (promedio polinomio)", "Constante (fijada)"],
    horizontal=True, index=0
)
eta_user = None
mode_key = "poly"
if eta_mode == "Constante (promedio polinomio)":
    mode_key = "poly-avg"
elif eta_mode == "Constante (fijada)":
    mode_key = "fixed"
    eta_user = st.number_input("η constante (0–1):", min_value=0.05, max_value=0.98, value=0.72, step=0.01)

# Mallado restringido al tramo elegido
N = 600
n_p_full = np.linspace(n_p_min, n_p_max, N)
mask = (n_p_full >= n_sel_min) & (n_p_full <= n_sel_max)
n_p = n_p_full[mask]
if n_p.size < 3:
    st.warning("Tramo demasiado estrecho; amplía el rango seleccionado.")
    n_p = np.linspace(n_sel_min, n_sel_max, 10)

# Cálculos
Q = k * n_p                        # m3/h
q = Q / 3600.0                     # m3/s
H = H0_m + K_m_s2 * (q ** 2)       # m
eta_grid = eta_eval(Q, row, mode_key, eta_user)
eta_min_used, eta_max_used, eta_mean_used = float(np.min(eta_grid)), float(np.max(eta_grid)), float(np.mean(eta_grid))

P_h = rho * g * q * H / np.maximum(eta_grid, 1e-6)  # W
omega_p = 2.0 * math.pi * n_p / 60.0                # rad/s
T_pump = np.where(omega_p > 1e-9, P_h / omega_p, 0.0)  # N·m
T_load_m = T_pump / max(r, 1e-9)                       # reflejado al motor
T_disp_m = np.full_like(T_load_m, T_nom_nm)

# Gráfico de Par vs n_p
fig_t = go.Figure()
fig_t.add_trace(go.Scatter(x=n_p, y=T_load_m, name="Par resistente reflejado (motor)", mode="lines"))
fig_t.add_trace(go.Scatter(x=n_p, y=T_disp_m, name="Par motor disponible", mode="lines"))
fig_t.update_layout(
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis_title="Velocidad bomba n_p [rpm]",
    yaxis_title="Par en el eje motor [N·m]",
    legend=dict(orientation="h", y=1.02, yanchor="bottom"),
    height=360,
)
st.plotly_chart(fig_t, use_container_width=True)

# Resumen de η usada
st.caption(
    f"**η usada** → min: {fmt_num(eta_min_used, '', 2)}, max: {fmt_num(eta_max_used, '', 2)}, media: {fmt_num(eta_mean_used, '', 2)}"
)

# Integración temporal con chequeo de par insuficiente
T_net = T_disp_m - T_load_m
if np.any(T_net <= 0):
    idx = int(np.argmax(T_net <= 0))
    st.error(
        f"Par motor insuficiente en ~{fmt_num(n_p[idx], 'rpm', 0)} (T_disp ≤ T_load). "
        f"No se puede acelerar en ese tramo; se omite el cálculo de tiempo hidráulico."
    )
    t_hyd = float("nan")
else:
    # integrar en el eje del motor
    n_m_seg = n_p * max(r, 1e-9)
    omega_m = 2.0 * math.pi * n_m_seg / 60.0
    d_omega = np.diff(omega_m)
    dt = (J_eq * d_omega) / T_net[:-1]
    t_hyd = float(np.sum(dt))

# Mostrar tiempo + “pill” debajo
c4_1, c4_2 = st.columns([1, 2])
with c4_1:
    st.markdown(f"- Tiempo por carga **hidráulica** (integración): {val_green(t_hyd, 's')}", unsafe_allow_html=True)
with c4_2:
    if np.isnan(t_hyd):
        pill("Tiempo limitante (sección 4): — (par insuficiente)", bg="#fdecea", color="#b71c1c")
    else:
        t_lim_4 = max(t_hyd, t_rampa) if not np.isnan(t_rampa) else t_hyd
        which = "hidráulica" if (np.isnan(t_rampa) or t_hyd > t_rampa) else "rampa VDF"
        pill(f"Tiempo limitante (sección 4): {which} = {fmt_num(t_lim_4, 's')}")

# Gráfico de integración: dt/dn_m y área bajo la curva
if not np.isnan(t_hyd):
    n_m_seg = n_p * max(r, 1e-9)
    dtdn = (J_eq * (2.0 * math.pi / 60.0)) / T_net  # s / rpm
    fig_dt = go.Figure()
    fig_dt.add_trace(go.Scatter(x=n_m_seg, y=dtdn, mode="lines", name="dt/dn_m"))
    fig_dt.add_trace(
        go.Scatter(
            x=np.r_[n_m_seg, n_m_seg[::-1]],
            y=np.r_[dtdn, np.zeros_like(dtdn)[::-1]],
            fill="toself",
            name="Área integrada (tiempo)",
            opacity=0.15,
        )
    )
    fig_dt.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="n_m [rpm]",
        yaxis_title="dt/dn_m [s/rpm]",
        legend=dict(orientation="h"),
        height=320,
    )
    st.plotly_chart(fig_dt, use_container_width=True)

with st.expander("Detalles y fórmulas — Sección 4"):
    st.markdown(
        r"""
- **Curva del sistema:** \(H(Q)=H_0+K\,(Q/3600)^2\) con \(K=\texttt{K\_m\_s2}\) del dataset.
- **Afinidad (25–50 Hz):** \(Q(n_p)=k\,n_p\) (sin término independiente). El tramo analizado es un subconjunto de \([n_{p,\min},n_{p,\max}]\).
- **Eficiencia:** si hay coeficientes, \(\eta(Q)=\eta_a+\eta_b\beta+\eta_c\beta^2\) con \(\beta=Q/Q_{\mathrm{ref}}\); si no, se usa \(\eta\) del dataset (o fija). Se acota a \([\eta_{\min},\eta_{\max}]\).
- **Potencia hidráulica:** \(P_h=\rho g\,Q_s\,H(Q)/\eta(Q)\), con \(Q_s=Q/3600\).
- **Par de bomba:** \(T_{\mathrm{pump}}=P_h/\omega_p\), \(\omega_p=2\pi n_p/60\). **Reflejo al motor:** \(T_{\mathrm{load,m}}=T_{\mathrm{pump}}/r\).
- **Par neto:** \(T_{\mathrm{net}}=T_{\mathrm{disp}}-T_{\mathrm{load,m}}\). Si \(T_{\mathrm{net}}\le 0\), no hay aceleración.
- **Integración temporal:** \(\Delta t=J_{eq}\,\Delta\omega_m/T_{\mathrm{net}}\). Continuo: \(\frac{dt}{dn_m}=\frac{J_{eq}}{T_{\mathrm{net}}}\frac{2\pi}{60}\); el tiempo es el área bajo esa curva.
        """
    )

st.markdown("---")

# =============================================================================
# 5) Exportar resultados del TAG a CSV
# =============================================================================
st.markdown("## 5) Exportar resultados del TAG a CSV")

calc = {
    "TAG": tag_sel,
    "pump_model": pump_model,
    "r_trans": r_nm_np,
    "n_m_min_rpm": n_m_min,
    "n_m_max_rpm": n_m_max,
    "n_p_min_rpm": n_p_min,
    "n_p_max_rpm": n_p_max,
    "P_motor_kW": P_motor_kW,
    "T_nom_nm": T_nom_nm,
    "rho_kgm3": rho,
    "H0_m": H0_m,
    "K_m_s2": K_m_s2,
    "k_Q_per_rpm": k,
    "J_m": J_m,
    "J_driver": J_driver,
    "J_bushing_driver": J_bushing_driver,
    "J_driven": J_driven,
    "J_bushing_driven": J_bushing_driven,
    "J_imp": J_imp,
    "J_eq": J_eq,
    "rampa_VDF_rpm_s": rampa_vdf,
    "n_dot_torque_rpm_s": n_dot_torque,
    "t_par_s": t_par,
    "t_rampa_s": t_rampa,
    "eta_mode": eta_mode,
    "eta_min_used": eta_min_used if 'eta_min_used' in locals() else np.nan,
    "eta_max_used": eta_max_used if 'eta_max_used' in locals() else np.nan,
    "eta_mean_used": eta_mean_used if 'eta_mean_used' in locals() else np.nan,
    "t_hidraulica_s": t_hyd if 't_hyd' in locals() else np.nan,
}

df_out = pd.DataFrame([calc])
csv_bytes = df_out.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Descargar CSV del TAG seleccionado",
    data=csv_bytes,
    file_name=f"{tag_sel}_resultado.csv",
    mime="text/csv",
)
