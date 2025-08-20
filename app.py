# -*- coding: utf-8 -*-
# Memoria de Cálculo – Tiempo de reacción de bombas (VDF)
# App Streamlit (sin Plotly). Lee: bombas_dataset_with_torque_params.xlsx en la raíz.

import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from io import StringIO

st.set_page_config(page_title="Memoria de Cálculo – Bombas (VDF)", layout="wide")

# ========= utilidades de estilo =========
PILL_CSS = """
<style>
.small-muted{color:#8aa;}
.pill {display:inline-block;padding:.25rem .6rem;border-radius:.6rem;border:1px solid #0a7f4555;
       background:#0a7f4516;color:#0a7f45;font-weight:800;}
.pill-red {display:inline-block;padding:.25rem .6rem;border-radius:.6rem;border:1px solid #b71c1c55;
       background:#b71c1c16;color:#b71c1c;font-weight:800;}
.h2{font-size:1.7rem;font-weight:900;margin:0.2rem 0 0.8rem 0;}
.ltx{font-size:1.05rem;}
.card {border:1px solid #e0e0e0; border-radius:12px; padding:12px 14px; margin-bottom:10px;}
.badge {font-weight:800;color:#344;padding:.15rem .45rem;border-radius:.5rem;background:#eef;border:1px solid #dde;}
hr{border:none;border-top:1px solid #e8e8e8;margin:10px 0;}
</style>
"""
st.markdown(PILL_CSS, unsafe_allow_html=True)

# ========= helpers generales =========

def norm_name(s: str) -> str:
    """normaliza nombres de columnas para matching robusto."""
    return ''.join(ch.lower() if ch.isalnum() else '_' for ch in str(s))

def find_col(df: pd.DataFrame, *candidatos) -> str:
    """
    Devuelve el nombre real de columna que mejor calza con cualquiera de los 'candidatos'
    (substrings normalizados). Lanza error si no encuentra.
    """
    cols_norm = {norm_name(c): c for c in df.columns}
    all_norm = list(cols_norm.keys())

    for cand in candidatos:
        nc = norm_name(cand)
        # match exact
        if nc in cols_norm:
            return cols_norm[nc]
        # match por "contiene"
        for k in all_norm:
            if nc in k:
                return cols_norm[k]
    raise KeyError(f"No se encontró columna para {candidatos}")

def to_num(x):
    """convierte texto con coma decimal a float."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace(' ', '')
    # coma decimal
    if ',' in s and '.' in s:
        # asume coma decimal y punto separador de miles
        s = s.replace('.', '').replace(',', '.')
    elif ',' in s:
        s = s.replace(',', '.')
    try:
        return float(s)
    except Exception:
        return np.nan

# ========= carga dataset raíz =========
@st.cache_data(show_spinner=False)
def load_dataset(path="bombas_dataset_with_torque_params.xlsx"):
    try:
        df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    except Exception:
        df = pd.read_excel(path, sheet_name=0)
    # normaliza por si vienen strings con coma
    for c in df.columns:
        df[c] = df[c].apply(to_num) if df[c].dtype == object else df[c]
    return df

df = load_dataset()

# Columna TAG (primera columna)
COL_TAG = df.columns[0]

# ===== mapeo robusto de columnas clave (según nombres que indicaste) =====
# transmisión / motor / bomba
COL_R             = find_col(df, "r_trans", "relaciontransmision", "r")
COL_T_NOM         = find_col(df, "t_nom_nm")
COL_J_MOTOR       = find_col(df, "motor_j_kgm2")
COL_J_IMP         = find_col(df, "impeller_j_kgm2")
COL_J_DRV_PUL     = find_col(df, "driverpulley_j_kgm2", "driver_pulley")
COL_J_DRV_BUSH    = find_col(df, "driverbushing_j_kgm2", "driver_bushing")
COL_J_DVN_PUL     = find_col(df, "drivenpulley_j_kgm2", "drivenpulley_j_kgm2", "drivenpulley_j_kgm2", "drivenpulley_j_kgm2".upper(), "drivenpulley_j_kgm2".title(), "drivenpulley")  # robust
COL_J_DVN_BUSH    = find_col(df, "drivenbushing_j_kgm2", "drivenbushing", "driven_bushing")
COL_NM_MIN        = find_col(df, "motor_n_min_rpm")
COL_NM_MAX        = find_col(df, "motor_n_max_rpm")
COL_NP_MIN        = find_col(df, "pump_n_min_rpm")
COL_NP_MAX        = find_col(df, "pump_n_max_rpm")

# hidráulica
COL_H0            = find_col(df, "H0_m")
COL_K             = find_col(df, "K_m_s2")
COL_R2_H          = find_col(df, "R2_H")
COL_ETA_A         = find_col(df, "eta_a")
COL_ETA_B         = find_col(df, "eta_b")
COL_ETA_C         = find_col(df, "eta_c")
COL_R2_ETA        = find_col(df, "R2_eta")
COL_Q_MIN         = find_col(df, "Q_min_m3h")
COL_Q_MAX         = find_col(df, "Q_max_m3h")
COL_Q_REF         = find_col(df, "Q_ref_m3h")
COL_N_REF         = find_col(df, "n_ref_rpm")
COL_RHO           = find_col(df, "rho_kgm3")
COL_ETA_BETA      = find_col(df, "eta_beta")
COL_ETA_MIN_CLIP  = find_col(df, "eta_min_clip")
COL_ETA_MAX_CLIP  = find_col(df, "eta_max_clip")

tags = df[COL_TAG].astype(str).tolist()

# ========= funciones de cálculo =========

def inercia_equivalente_row(row):
    """J_eq al eje del motor (kg·m²). r = n_motor/n_bomba."""
    r = max(1e-9, to_num(row[COL_R]))
    Jm   = max(0.0, to_num(row[COL_J_MOTOR]))
    Jimp = max(0.0, to_num(row[COL_J_IMP]))
    Jdrv = max(0.0, to_num(row[COL_J_DRV_PUL])) + max(0.0, to_num(row[COL_J_DRV_BUSH]))
    Jdvn = max(0.0, to_num(row[COL_J_DVN_PUL])) + max(0.0, to_num(row[COL_J_DVN_BUSH]))
    # Reflejo del lado bomba al motor:
    J_eq = Jm + Jdrv + (Jdvn + Jimp) / (r**2)
    return J_eq, dict(Jm=Jm, Jimp=Jimp, Jdrv=Jdrv, Jdvn=Jdvn, r=r)

def n_dot_por_torque(T_disp_Nm, J_eq_kgm2):
    """tasa de aceleración debida a par [rpm/s] (al eje del motor)."""
    if J_eq_kgm2 <= 0: 
        return np.nan
    return (60.0/(2*math.pi)) * (T_disp_Nm / J_eq_kgm2)

def eta_poly(Q, Qref, a, b, c, eta_min, eta_max):
    """η = clip(a + b*(Q/Qref) + c*(Q/Qref)^2, [eta_min, eta_max])."""
    x = 0.0 if Qref<=0 else (Q / Qref)
    eta = a + b*x + c*(x**2)
    if eta_min is not None:
        eta = max(eta_min, eta)
    if eta_max is not None:
        eta = min(eta_max, eta)
    return max(1e-6, eta)

def H_system(Q_m3h, H0, K):
    """H(Q) = H0 + K * (Q/3600)^2"""
    return H0 + K * ( (Q_m3h/3600.0)**2 )

def torque_pump(Q_m3h, n_p_rpm, rho, H0, K, eta):
    """Par en eje de bomba (Nm)."""
    if n_p_rpm <= 0:
        return np.inf  # sin velocidad -> omega=0 => par requerido "infinito" si hay carga
    H = H_system(Q_m3h, H0, K)
    Qs = Q_m3h/3600.0
    Ph = rho * 9.80665 * Qs * H / max(1e-6, eta)  # W
    omega_p = 2*math.pi*n_p_rpm/60.0
    return Ph / omega_p  # Nm

def simulate_hydraulics_trajectory(row, n_p_ini, n_p_fin, ramp_motor_rps, overrides=None, dt=0.01):
    """
    Integración simple forward-Euler:
    - n_motor_dot limitada por:  (i) torque disponible - torque hidráulico reflejado, (ii) rampa del VDF.
    - Q ~ n_p (afinidad), se usa Q_ref/n_ref para la proporcionalidad.
    Retorna dict con arrays y t_total (NaN si no converge).
    """
    if overrides is None: overrides = {}

    r   = max(1e-9, to_num(row[COL_R]))
    J_eq,_  = inercia_equivalente_row(row)
    T_disp  = max(0.0, to_num(row[COL_T_NOM]))

    # hidráulica efectivas
    H0  = overrides.get("H0",  to_num(row[COL_H0]))
    K   = overrides.get("K",   to_num(row[COL_K]))
    rho = overrides.get("rho", to_num(row[COL_RHO]))
    a   = overrides.get("eta_a", to_num(row[COL_ETA_A]))
    b   = overrides.get("eta_b", to_num(row[COL_ETA_B]))
    c   = overrides.get("eta_c", to_num(row[COL_ETA_C]))
    eta_min = to_num(row[COL_ETA_MIN_CLIP]); eta_max = to_num(row[COL_ETA_MAX_CLIP])
    Qref = max(1e-9, to_num(row[COL_Q_REF]))
    nref = max(1e-9, to_num(row[COL_N_REF]))

    # rampa en eje bomba (rpm/s)
    ramp_pump_rps = ramp_motor_rps / max(1e-9, r)

    # estado
    n_p = float(n_p_ini)
    n_m = n_p * r
    t   = 0.0

    arr_t, arr_np, arr_Q, arr_Ph = [], [], [], []

    # si par de arranque imposible, abortar
    # (aprox Q ~ (Qref/nref)*n_p; para n_p ~ 0, usamos un np pequeño)
    eps_np = max(1.0, 0.01 * nref)
    Q0 = Qref/nref * max(eps_np, n_p)
    eta0 = eta_poly(Q0, Qref, a,b,c, eta_min, eta_max)
    T_p0 = torque_pump(Q0, max(eps_np, n_p), rho, H0, K, eta0)
    T_ref0 = T_p0 / r
    if T_disp <= T_ref0 and n_p_ini <= 0.1:
        return dict(t=np.array(arr_t), n_p=np.array(arr_np), Q=np.array(arr_Q), Ph=np.array(arr_Ph),
                    t_total=np.nan)

    # integrar
    max_steps = int(1e7)  # guard
    steps = 0
    while n_p < n_p_fin and steps < max_steps:
        steps += 1

        # caudal por afinidad
        Q = Qref/nref * n_p
        eta = eta_poly(Q, Qref, a,b,c, eta_min, eta_max)
        T_p = torque_pump(Q, max(1e-6, n_p), rho, H0, K, eta)
        T_ref = T_p / r

        # aceleración por torque al motor
        n_dot_torque_motor = (60.0/(2*math.pi)) * ( (T_disp - T_ref) / max(1e-9, J_eq) )
        # limitado por rampa del VDF
        n_dot_motor = min(n_dot_torque_motor, ramp_motor_rps)
        # si ya no hay par suficiente (n_dot<=0) -> no converge para ese rango
        if n_dot_motor <= 0:
            return dict(t=np.array(arr_t), n_p=np.array(arr_np), Q=np.array(arr_Q), Ph=np.array(arr_Ph),
                        t_total=np.nan)

        # integrar
        n_m += n_dot_motor * dt
        n_p  = n_m / r
        t   += dt

        Ph = rho * 9.80665 * (Q/3600.0) * H_system(Q, H0, K) / max(1e-6, eta)

        # guardar
        arr_t.append(t); arr_np.append(n_p); arr_Q.append(Q); arr_Ph.append(Ph)

        # safety: si tarda demasiado para rangos chicos
        if t > 1e4:  # 2h+ de simulación
            break

    return dict(t=np.array(arr_t), n_p=np.array(arr_np), Q=np.array(arr_Q), Ph=np.array(arr_Ph),
                t_total=(t if len(arr_t)>0 and n_p>=n_p_fin-1e-6 else np.nan))

# ========= tarjetas compactas de ajuste (±30% paso 1%) =========
ADJ_CSS = """
<style>
.ctrl-card {border:1px solid #cfd8dc55; border-radius:12px; padding:10px 12px; margin-bottom:12px;}
.ctrl-title {font-weight:700; font-size:0.92rem;}
.ctrl-buttons button {width:42px;height:36px;padding:0;margin:0 2px;}
</style>
"""
st.markdown(ADJ_CSS, unsafe_allow_html=True)

def percent_card(key, label, base_value, unit=""):
    skey = f"pct_{key}_{selected}"
    if skey not in st.session_state:
        st.session_state[skey] = 100  # %
    with st.container():
        st.markdown('<div class="ctrl-card">', unsafe_allow_html=True)
        c1, c2 = st.columns([0.60, 0.40])
        with c1:
            st.markdown(f'<div class="ctrl-title">{label} (±30%)</div>', unsafe_allow_html=True)
            st.session_state[skey] = st.slider(
                f"ajuste_{skey}", min_value=70, max_value=130, step=1,
                value=st.session_state[skey], label_visibility="collapsed")
        with c2:
            bcol1, bcol2, bcol3 = st.columns([1,1,1])
            with bcol1:
                if st.button("−1%", key=f"dec_{skey}"):
                    st.session_state[skey] = max(70, st.session_state[skey]-1)
            with bcol2:
                if st.button("Reset", key=f"res_{skey}"):
                    st.session_state[skey] = 100
            with bcol3:
                if st.button("+1%", key=f"inc_{skey}"):
                    st.session_state[skey] = min(130, st.session_state[skey]+1)
            eff = base_value * (st.session_state[skey]/100)
            st.markdown(f'<div class="pill">{eff:.2f} {unit}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    return eff

# ========= barra lateral =========
st.sidebar.header("Selección")
selected = st.sidebar.selectbox("TAG", options=tags, index=0)
row = df[df[COL_TAG].astype(str) == str(selected)].iloc[0]

# ========= Sección 1: parámetros de entrada (Motor, Transmisión, Bomba) =========
st.markdown('<div class="h2">1) Parámetros de entrada</div>', unsafe_allow_html=True)

J_eq, parts = inercia_equivalente_row(row)
nm_min = to_num(row[COL_NM_MIN]); nm_max = to_num(row[COL_NM_MAX])
np_min = to_num(row[COL_NP_MIN]); np_max = to_num(row[COL_NP_MAX])
T_nom  = to_num(row[COL_T_NOM])

colM, colT, colB = st.columns([1,1,1])

with colM:
    st.subheader("Motor")
    st.markdown("Potencia/Par nominal")
    st.markdown(f'<span class="pill">{T_nom:.2f} Nm</span>', unsafe_allow_html=True)
    st.markdown("Velocidad Motor min–max [rpm]")
    st.markdown(f'<span class="pill">{nm_min:.0f} – {nm_max:.0f}</span>', unsafe_allow_html=True)
    st.markdown("J_m (kg·m²)")
    st.markdown(f'<span class="pill">{parts["Jm"]:.2f}</span>', unsafe_allow_html=True)

with colT:
    st.subheader("Transmisión")
    st.markdown("Relación r = n_motor/n_bomba")
    st.markdown(f'<span class="pill">{parts["r"]:.2f}</span>', unsafe_allow_html=True)
    st.markdown("J_driver total (kg·m²)")
    st.markdown(f'<span class="pill">{parts["Jdrv"]:.2f}</span>', unsafe_allow_html=True)
    st.markdown("J_driven total (kg·m²)")
    st.markdown(f'<span class="pill">{parts["Jdvn"]:.2f}</span>', unsafe_allow_html=True)

with colB:
    st.subheader("Bomba")
    st.markdown("Velocidad Bomba min–max [rpm]")
    st.markdown(f'<span class="pill">{np_min:.0f} – {np_max:.0f}</span>', unsafe_allow_html=True)
    st.markdown("J_imp (kg·m²)")
    st.markdown(f'<span class="pill">{parts["Jimp"]:.2f}</span>', unsafe_allow_html=True)
    st.markdown("J_eq (kg·m²) al eje del motor")
    st.markdown(f'<span class="pill">{J_eq:.2f}</span>', unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# ========= Sección 2: Inercia equivalente al eje del motor (con fórmula) =========
st.markdown('<div class="h2">2) Inercia equivalente al eje del motor</div>', unsafe_allow_html=True)

st.latex(r"J_{\text{eq}} \;=\; J_m \;+\; J_{\text{driver}}\;+\;\dfrac{J_{\text{driven}} + J_{\text{imp}}}{r^2}")

st.markdown(
    f"**Sustitución numérica:**  \n"
    r"$J_{\text{eq}} = J_m + J_{\text{driver}} + \dfrac{J_{\text{driven}} + J_{\text{imp}}}{r^2} \;=\;$ "
    f"{parts['Jm']:.2f} + {parts['Jdrv']:.2f} + "
    r"\dfrac{" + f"{parts['Jdvn']:.2f} + {parts['Jimp']:.2f}" + r"}{" + f"({parts['r']:.2f})^2" + r"}"
)
st.markdown(f'<div class="pill" style="margin-top:6px;">J_eq = {J_eq:.2f} kg·m²</div>', unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# ========= Sección 3: Respuesta inercial (sin efectos hidráulicos) =========
st.markdown('<div class="h2">3) Respuesta inercial (sin efectos hidráulicos)</div>', unsafe_allow_html=True)

c3a, c3b, c3c = st.columns(3)
with c3a:
    n_motor_ini = st.number_input("Velocidad Motor inicial [rpm]", value=float(nm_min), step=1.0, format="%.2f")
with c3b:
    n_motor_fin = st.number_input("Velocidad Motor final [rpm]", value=float(nm_max), step=1.0, format="%.2f")
with c3c:
    T_disp_ui   = st.number_input("Par disponible T_nom [Nm]", value=float(T_nom), step=1.0, format="%.2f")

st.latex(r"\dot n_{\text{torque}} \;=\; \dfrac{60}{2\pi}\,\dfrac{T_{\text{disp}}}{J_{\text{eq}}} \quad;\quad "
         r"t_{\text{par}}=\dfrac{\Delta n}{\dot n_{\text{torque}}} \quad;\quad "
         r"t_{\text{rampa}}=\dfrac{\Delta n}{\text{rampa}_{\text{VDF}}} \quad;\quad "
         r"t_{\text{final,sin}}=\max(t_{\text{par}},t_{\text{rampa}})")

c3d, c3e = st.columns([1,1])
with c3d:
    ramp_motor = st.slider("Rampa del VDF (rpm/s en motor)", min_value=50, max_value=1000, step=10, value=300)
with c3e:
    st.write("")  # spacing

delta_n = max(0.0, n_motor_fin - n_motor_ini)
n_dot_tq = n_dot_por_torque(T_disp_ui, J_eq)
t_par    = delta_n / max(1e-9, n_dot_tq) if not np.isnan(n_dot_tq) else np.nan
t_rampa  = delta_n / max(1e-9, ramp_motor)
t_final_sin = max(t_par, t_rampa) if not (np.isnan(t_par) or np.isnan(t_rampa)) else np.nan

c3x, c3y, c3z, c3w = st.columns(4)
c3x.markdown(f'<div class="pill">Δn = {delta_n:.2f} rpm</div>', unsafe_allow_html=True)
c3y.markdown(f'<div class="pill">\\(\\dot n_{{\\mathrm{{torque}}}}\\) = {n_dot_tq:.2f} rpm/s</div>', unsafe_allow_html=True)
c3z.markdown(f'<div class="pill">t_{{\\mathrm{{par}}}} = {t_par:.2f} s</div>', unsafe_allow_html=True)
c3w.markdown(f'<div class="pill">t_{{\\mathrm{{rampa}}}} = {t_rampa:.2f} s</div>', unsafe_allow_html=True)
st.markdown(f'<div class="pill" style="font-size:1.05rem;margin-top:6px;">t_{{\\mathrm{{final,sin}}}} = {t_final_sin:.2f} s</div>',
            unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# ========= Sección 4: Sistema (H–Q, η) y densidad – ajustes y respuesta con hidráulica =========
st.markdown('<div class="h2">4) Sistema (H–Q, η) y densidad – ajustes y respuesta con hidráulica</div>', unsafe_allow_html=True)
st.caption("Ajustes relativos ±30% con paso 1%. Los valores efectivos afectan la integración.")

# tarjetas compactas de ajuste
cS1, cS2, cS3 = st.columns(3)
with cS1:
    H0_eff   = percent_card("H0",   "H0 [m]",            to_num(row[COL_H0]),   "m")
    eta_a_eff= percent_card("eta_a","η_a [−]",           to_num(row[COL_ETA_A]),  "")
with cS2:
    K_eff    = percent_card("K",    "K [m·s²/m⁶]",       to_num(row[COL_K]), "m·s²/m⁶")
    eta_b_eff= percent_card("eta_b","η_b [−]",           to_num(row[COL_ETA_B]),  "")
with cS3:
    rho_eff  = percent_card("rho",  "ρ pulpa [kg/m³]",   to_num(row[COL_RHO]), "kg/m³")
    eta_c_eff= percent_card("eta_c","η_c [−]",           to_num(row[COL_ETA_C]),  "")

overrides = dict(H0=H0_eff, K=K_eff, eta_a=eta_a_eff, eta_b=eta_b_eff, eta_c=eta_c_eff, rho=rho_eff)

# rango de bomba a evaluar: 0 rpm -> máx (a 50 Hz, viene en dataset)
n_p_ini_sel = st.slider("Rango de BOMBA a evaluar [rpm] (inicio → fin)",
                        min_value=0, max_value=int(np_max), value=(0, int(np_max)))[0]
# Para usar un extremo: de 0 hasta el máximo
n_p_ini_sel = 0
n_p_fin_sel = int(np_max)

st.caption(f"Rampa del VDF (motor) usada (de 3): {ramp_motor} rpm/s")

traj = simulate_hydraulics_trajectory(row, n_p_ini_sel, n_p_fin_sel, ramp_motor, overrides=overrides)

if np.isnan(traj["t_total"]):
    if traj["n_p"].size:
        n_last = traj["n_p"][-1]; Q_last = traj["Q"][-1]
        st.error(f"El cálculo hidráulico no converge (par disponible insuficiente para ese rango). "
                 f"Se detuvo cerca de n_bomba ≈ {n_last:.0f} rpm, Q ≈ {Q_last:.0f} m³/h.")
    else:
        st.error("El cálculo hidráulico no converge desde el inicio (par de arranque insuficiente).")
else:
    # Diagnóstico de limitante
    # Tiempo por rampa sólo (en bomba, equivalente en motor)
    delta_n_motor_tot = (n_p_fin_sel - n_p_ini_sel) * parts["r"]
    t_rampa_only = delta_n_motor_tot / max(1e-9, ramp_motor)
    t_hid = traj["t_total"]
    limitante = "VDF (rampa)" if t_rampa_only >= t_hid else "Par hidráulico"

    c4a, c4b, c4c = st.columns(3)
    c4a.markdown(f'<div class="pill">t_{{\\mathrm{{hidráulica}}}} = {t_hid:.2f} s</div>', unsafe_allow_html=True)
    c4b.markdown(f'<div class="pill">t_{{\\mathrm{{rampa}}}} = {t_rampa_only:.2f} s</div>', unsafe_allow_html=True)
    c4c.markdown(f'<div class="pill">Limitante: {limitante}</div>', unsafe_allow_html=True)

    # -------- gráfico Q(t), n_p(t) y P_h(t) (kW) --------
    fig, ax1 = plt.subplots(figsize=(8.0, 3.4), dpi=140)
    t = traj["t"]
    ax1.plot(t, traj["Q"], label="Q (m³/h)")
    ax1.set_xlabel("t (s)")
    ax1.set_ylabel("Q (m³/h)")
    ax1.grid(True, alpha=.25)

    ax2 = ax1.twinx()
    ax2.plot(t, traj["n_p"], linestyle="--", label="n_bomba (rpm)")
    ax2.plot(t, np.array(traj["Ph"])/1000.0, linestyle=":", label="P_h (kW)")
    ax2.set_ylabel("n_bomba (rpm) / P_h (kW)")

    # leyenda combinada
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper left", fontsize=8)
    st.pyplot(fig, clear_figure=True)

# ========= Exportación de tabla por TAG (con rampa seleccionada) =========
st.markdown("<hr/>", unsafe_allow_html=True)
st.subheader("Exportar resumen por TAG (rampa actual)")

def resumen_por_tag(row, ramp_motor):
    J_eq, parts = inercia_equivalente_row(row)
    nm_min = to_num(row[COL_NM_MIN]); nm_max = to_num(row[COL_NM_MAX])
    np_max = to_num(row[COL_NP_MAX])
    T_nom  = to_num(row[COL_T_NOM])
    # 3) sin hidráulica
    delta_n = max(0.0, nm_max - nm_min)
    n_dot_tq = n_dot_por_torque(T_nom, J_eq)
    t_par    = delta_n / max(1e-9, n_dot_tq) if not np.isnan(n_dot_tq) else np.nan
    t_rampa  = delta_n / max(1e-9, ramp_motor)
    t_final_sin = max(t_par, t_rampa) if not (np.isnan(t_par) or np.isnan(t_rampa)) else np.nan
    # 4) hidráulica 0→n_p_max
    traj = simulate_hydraulics_trajectory(row, 0, np_max, ramp_motor, overrides=None)
    t_hid = traj["t_total"]
    # comparación para rango hidráulico
    delta_n_motor_tot = (np_max - 0) * parts["r"]
    t_rampa_only = delta_n_motor_tot / max(1e-9, ramp_motor)
    limit = "no converge"
    if not np.isnan(t_hid):
        limit = "VDF (rampa)" if t_rampa_only >= t_hid else "Par hidráulico"
    return {
        "TAG": str(row[COL_TAG]),
        "J_eq_kgm2": round(J_eq, 3),
        "n_dot_torque_rpm_s": round(n_dot_tq, 2) if not np.isnan(n_dot_tq) else np.nan,
        "t_par_sin_s": round(t_par, 2) if not np.isnan(t_par) else np.nan,
        "t_rampa_sin_s": round(t_rampa, 2),
        "t_final_sin_s": round(t_final_sin, 2) if not np.isnan(t_final_sin) else np.nan,
        "t_hidraulica_0_a_npmax_s": round(t_hid, 2) if not np.isnan(t_hid) else np.nan,
        "limitante_hidraulica": limit
    }

if st.button("Generar CSV"):
    rows = [resumen_por_tag(df.iloc[i], ramp_motor) for i in range(len(df))]
    out = pd.DataFrame(rows)
    csv = out.to_csv(index=False)
    st.download_button("Descargar resumen.csv", csv, file_name="resumen_por_tag.csv", mime="text/csv")
