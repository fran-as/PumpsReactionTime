# -*- coding: utf-8 -*-
# Memoria de Cálculo – Tiempo de reacción de bombas (VDF)
# Lee bombas_dataset_with_torque_params.xlsx desde la raíz

import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Memoria de Cálculo – Bombas (VDF)", layout="wide")

# -------------------- Estilos mínimos --------------------
st.markdown("""
<style>
.small-muted{color:#8aa;}
.pill{display:inline-block;padding:.25rem .6rem;border-radius:.6rem;border:1px solid #0a7f4555;
      background:#0a7f4516;color:#0a7f45;font-weight:800}
.card{border:1px solid #e9eef2;border-radius:12px;padding:12px;margin-bottom:10px}
.h2{font-size:1.7rem;font-weight:900;margin:.2rem 0 .8rem 0}
hr{border:none;border-top:1px solid #eee;margin:12px 0}
</style>
""", unsafe_allow_html=True)

# -------------------- Utils --------------------
def norm_name(s: str) -> str:
    return ''.join(ch.lower() if ch.isalnum() else '_' for ch in str(s))

def find_col(df: pd.DataFrame, *cands) -> str:
    norm = {norm_name(c): c for c in df.columns}
    keys = list(norm.keys())
    for c in cands:
        nc = norm_name(c)
        if nc in norm:
            return norm[nc]
        for k in keys:
            if nc in k:
                return norm[k]
    raise KeyError(f"Columna no encontrada para: {cands}")

def to_num(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float, np.number)): return float(x)
    s = str(x).strip().replace(' ', '').replace('\u00a0','')
    # coma decimal -> punto
    if ',' in s and '.' in s:
        s = s.replace('.', '').replace(',', '.')
    elif ',' in s:
        s = s.replace(',', '.')
    try:
        return float(s)
    except Exception:
        return np.nan

# -------------------- Cargar dataset --------------------
@st.cache_data(show_spinner=False)
def load_dataset(path="bombas_dataset_with_torque_params.xlsx"):
    try:
        df = pd.read_excel(path, engine="openpyxl")
    except Exception:
        df = pd.read_excel(path)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].apply(to_num)
    return df

df = load_dataset()
TAG_COL = df.columns[0]

# --- nombres de columnas (tal como indicaste) ---
COL_R            = find_col(df, "r_trans")
COL_T_NOM        = find_col(df, "t_nom_nm")
COL_J_MOTOR      = find_col(df, "motor_j_kgm2")
COL_J_IMP        = find_col(df, "impeller_j_kgm2")
COL_J_DRV_PUL    = find_col(df, "driverpulley_j_kgm2")
COL_J_DRV_BUSH   = find_col(df, "driverbushing_j_kgm2")
COL_J_DVN_PUL    = find_col(df, "drivenpulley_j_kgm2", "drivenpulley_j_kgm2", "drivenpulley_j_kgm2".upper(), "driven_pulley")
COL_J_DVN_BUSH   = find_col(df, "drivenbushing_j_kgm2", "driven_bushing")
COL_NM_MIN       = find_col(df, "motor_n_min_rpm")
COL_NM_MAX       = find_col(df, "motor_n_max_rpm")
COL_NP_MIN       = find_col(df, "pump_n_min_rpm")
COL_NP_MAX       = find_col(df, "pump_n_max_rpm")
COL_H0           = find_col(df, "H0_m")
COL_K            = find_col(df, "K_m_s2")
COL_R2_H         = find_col(df, "R2_H")
COL_ETA_A        = find_col(df, "eta_a")
COL_ETA_B        = find_col(df, "eta_b")
COL_ETA_C        = find_col(df, "eta_c")
COL_R2_ETA       = find_col(df, "R2_eta")
COL_Q_MIN        = find_col(df, "Q_min_m3h")
COL_Q_MAX        = find_col(df, "Q_max_m3h")
COL_Q_REF        = find_col(df, "Q_ref_m3h")
COL_N_REF        = find_col(df, "n_ref_rpm")
COL_RHO          = find_col(df, "rho_kgm3")
COL_ETA_BETA     = find_col(df, "eta_beta")
COL_ETA_MIN      = find_col(df, "eta_min_clip")
COL_ETA_MAX      = find_col(df, "eta_max_clip")

tags = df[TAG_COL].astype(str).tolist()

# -------------------- Modelo --------------------
def inercia_eq(row):
    r = max(1e-9, to_num(row[COL_R]))
    Jm   = max(0.0, to_num(row[COL_J_MOTOR]))
    Jimp = max(0.0, to_num(row[COL_J_IMP]))
    Jdrv = max(0.0, to_num(row[COL_J_DRV_PUL]))  + max(0.0, to_num(row[COL_J_DRV_BUSH]))
    Jdvn = max(0.0, to_num(row[COL_J_DVN_PUL])) + max(0.0, to_num(row[COL_J_DVN_BUSH]))
    return Jm + Jdrv + (Jdvn + Jimp) / (r**2), dict(Jm=Jm, Jimp=Jimp, Jdrv=Jdrv, Jdvn=Jdvn, r=r)

def n_dot_torque(T_disp, J_eq):
    return (60.0/(2*math.pi)) * (T_disp / max(1e-9, J_eq))

def eta_poly(Q, Qref, a,b,c, emin, emax):
    x = 0.0 if Qref<=0 else (Q/Qref)
    eta = a + b*x + c*x*x
    if emin is not None: eta = max(emin, eta)
    if emax is not None: eta = min(emax, eta)
    return max(1e-6, eta)

def H_sys(Q_m3h, H0, K):
    return H0 + K * ((Q_m3h/3600.0)**2)

def torque_pump(Q_m3h, n_p_rpm, rho, H0, K, eta):
    if n_p_rpm <= 0: return np.inf
    H = H_sys(Q_m3h, H0, K)
    Qs = Q_m3h/3600.0
    Ph = rho*9.80665*Qs*H/max(1e-6, eta)  # W
    omega_p = 2*math.pi*n_p_rpm/60.0
    return Ph / omega_p  # Nm

def simulate(row, n_p_ini, n_p_fin, ramp_motor_rps, overrides=None, dt=0.01):
    if overrides is None: overrides = {}
    r = max(1e-9, to_num(row[COL_R]))
    J_eq,_ = inercia_eq(row)
    T_disp = max(0.0, to_num(row[COL_T_NOM]))

    H0  = overrides.get("H0",  to_num(row[COL_H0]))
    K   = overrides.get("K",   to_num(row[COL_K]))
    rho = overrides.get("rho", to_num(row[COL_RHO]))
    a   = overrides.get("eta_a", to_num(row[COL_ETA_A]))
    b   = overrides.get("eta_b", to_num(row[COL_ETA_B]))
    c   = overrides.get("eta_c", to_num(row[COL_ETA_C]))
    emin = to_num(row[COL_ETA_MIN]); emax = to_num(row[COL_ETA_MAX])
    Qref = max(1e-9, to_num(row[COL_Q_REF]))
    nref = max(1e-9, to_num(row[COL_N_REF]))

    ramp_p = ramp_motor_rps / r

    n_p = float(n_p_ini)
    n_m = n_p * r
    t   = 0.0
    tt, NP, QQ, PH = [], [], [], []

    # chequeo arranque
    eps = max(1.0, 0.01*nref)
    Q0 = Qref/nref * max(eps, n_p)
    eta0 = eta_poly(Q0, Qref, a,b,c,emin,emax)
    T_p0 = torque_pump(Q0, max(eps, n_p), rho, H0, K, eta0)
    if T_disp <= T_p0/r and n_p_ini <= 0.1:
        return dict(t=np.array(tt), n_p=np.array(NP), Q=np.array(QQ), Ph=np.array(PH), t_total=np.nan)

    steps=0; MAX=10_000_000
    while n_p < n_p_fin and steps<MAX:
        steps+=1
        Q = Qref/nref * n_p
        eta = eta_poly(Q, Qref, a,b,c,emin,emax)
        T_p = torque_pump(Q, max(1e-6,n_p), rho, H0, K, eta)
        T_ref = T_p / r

        n_dot_motor = min(n_dot_torque(T_disp - T_ref, J_eq), ramp_motor_rps)
        if n_dot_motor <= 0:
            return dict(t=np.array(tt), n_p=np.array(NP), Q=np.array(QQ), Ph=np.array(PH), t_total=np.nan)

        n_m += n_dot_motor*dt
        n_p  = n_m / r
        t   += dt

        Ph = rho*9.80665*(Q/3600.0)*H_sys(Q, H0, K)/max(1e-6, eta)
        tt.append(t); NP.append(n_p); QQ.append(Q); PH.append(Ph)
        if t>36000: break

    return dict(t=np.array(tt), n_p=np.array(NP), Q=np.array(QQ), Ph=np.array(PH),
                t_total=(t if len(tt)>0 and n_p>=n_p_fin-1e-6 else np.nan))

# -------------------- UI --------------------
st.sidebar.header("Selección")
selected = st.sidebar.selectbox("TAG", options=tags, index=0)
row = df[df[TAG_COL].astype(str)==str(selected)].iloc[0]

st.markdown('<div class="h2">1) Parámetros de entrada</div>', unsafe_allow_html=True)

J_eq, parts = inercia_eq(row)
nm_min = to_num(row[COL_NM_MIN]); nm_max = to_num(row[COL_NM_MAX])
np_min = to_num(row[COL_NP_MIN]); np_max = to_num(row[COL_NP_MAX])
T_nom  = to_num(row[COL_T_NOM])

c1,c2,c3 = st.columns(3)
with c1:
    st.subheader("Motor")
    st.write("Par nominal [Nm]"); st.markdown(f'<span class="pill">{T_nom:.2f}</span>', unsafe_allow_html=True)
    st.write("Velocidad Motor [rpm]"); st.markdown(f'<span class="pill">{nm_min:.0f} – {nm_max:.0f}</span>', unsafe_allow_html=True)
    st.write("J_m [kg·m²]"); st.markdown(f'<span class="pill">{parts["Jm"]:.2f}</span>', unsafe_allow_html=True)
with c2:
    st.subheader("Transmisión")
    st.write("Relación r=n_m/n_p"); st.markdown(f'<span class="pill">{parts["r"]:.2f}</span>', unsafe_allow_html=True)
    st.write("J_driver [kg·m²]"); st.markdown(f'<span class="pill">{parts["Jdrv"]:.2f}</span>', unsafe_allow_html=True)
    st.write("J_driven [kg·m²]"); st.markdown(f'<span class="pill">{parts["Jdvn"]:.2f}</span>', unsafe_allow_html=True)
with c3:
    st.subheader("Bomba")
    st.write("Velocidad Bomba [rpm]"); st.markdown(f'<span class="pill">{np_min:.0f} – {np_max:.0f}</span>', unsafe_allow_html=True)
    st.write("J_imp [kg·m²]"); st.markdown(f'<span class="pill">{parts["Jimp"]:.2f}</span>', unsafe_allow_html=True)
    st.write("J_eq [kg·m²]"); st.markdown(f'<span class="pill">{J_eq:.2f}</span>', unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

st.markdown('<div class="h2">2) Inercia equivalente al eje del motor</div>', unsafe_allow_html=True)
st.latex(r"J_{\mathrm{eq}} = J_m + J_{\mathrm{driver}} + \dfrac{J_{\mathrm{driven}}+J_{\mathrm{imp}}}{r^2}")
st.markdown(f"**Sustitución:** {parts['Jm']:.2f} + {parts['Jdrv']:.2f} + "
            f"( {parts['Jdvn']:.2f} + {parts['Jimp']:.2f} ) / ({parts['r']:.2f})² "
            f"→ **J_eq** = {J_eq:.2f} kg·m²")

st.markdown("<hr/>", unsafe_allow_html=True)

st.markdown('<div class="h2">3) Respuesta inercial (sin efectos hidráulicos)</div>', unsafe_allow_html=True)
c31,c32,c33 = st.columns(3)
with c31:
    n_m_ini = st.number_input("Vel. Motor inicial [rpm]", value=float(nm_min), step=1.0, format="%.2f")
with c32:
    n_m_fin = st.number_input("Vel. Motor final [rpm]", value=float(nm_max), step=1.0, format="%.2f")
with c33:
    ramp_motor = st.slider("Rampa VDF [rpm/s]", min_value=50, max_value=1000, step=10, value=300)

delta_n = max(0.0, n_m_fin - n_m_ini)
n_dot   = n_dot_torque(T_nom, J_eq)
t_par   = delta_n / max(1e-9, n_dot)
t_rampa = delta_n / max(1e-9, ramp_motor)
t_final_sin = max(t_par, t_rampa)

c34,c35,c36,c37 = st.columns(4)
c34.markdown(f'<span class="pill">Δn = {delta_n:.2f} rpm</span>', unsafe_allow_html=True)
c35.markdown(f'<span class="pill">\\(\\dot n_{{torque}}\\) = {n_dot:.2f} rpm/s</span>', unsafe_allow_html=True)
c36.markdown(f'<span class="pill">t_par = {t_par:.2f} s</span>', unsafe_allow_html=True)
c37.markdown(f'<span class="pill">t_rampa = {t_rampa:.2f} s</span>', unsafe_allow_html=True)
st.markdown(f'<span class="pill" style="font-size:1.05rem;">t_final,sin = {t_final_sin:.2f} s</span>', unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)

# -------- Sección 4: Sistema (H–Q, η) + densidad; ajustes y simulación hidráulica --------
st.markdown('<div class="h2">4) Sistema (H–Q, η) y densidad – ajustes y respuesta con hidráulica</div>', unsafe_allow_html=True)
st.caption("Ajustes relativos sobre el dataset (±30%) con paso 1%. Los valores efectivos afectan la integración.")

def pct_control(key, label, base, unit=""):
    skey = f"{key}_{selected}"
    if skey not in st.session_state: st.session_state[skey] = 100
    colA,colB = st.columns([0.7,0.3])
    with colA:
        st.write(label + " (±30%)")
        st.session_state[skey] = st.slider("aj_"+skey, 70, 130, st.session_state[skey], 1, label_visibility="collapsed")
    with colB:
        b1,b2,b3 = st.columns(3)
        with b1:
            if st.button("−1%", key="d"+skey): st.session_state[skey] = max(70, st.session_state[skey]-1)
        with b2:
            if st.button("Reset", key="r"+skey): st.session_state[skey] = 100
        with b3:
            if st.button("+1%", key="i"+skey): st.session_state[skey] = min(130, st.session_state[skey]+1)
        eff = base * (st.session_state[skey]/100.0)
        st.markdown(f'<span class="pill">{eff:.2f} {unit}</span>', unsafe_allow_html=True)
    return eff

s1,s2,s3 = st.columns(3)
with s1:
    H0_eff   = pct_control("H0",   "H0 [m]",          to_num(row[COL_H0]),   "m")
    eta_a    = pct_control("ea",   "η_a [−]",         to_num(row[COL_ETA_A]), "")
with s2:
    K_eff    = pct_control("K",    "K [m·s²/m⁶]",     to_num(row[COL_K]),    "m·s²/m⁶")
    eta_b    = pct_control("eb",   "η_b [−]",         to_num(row[COL_ETA_B]), "")
with s3:
    rho_eff  = pct_control("rho",  "ρ pulpa [kg/m³]", to_num(row[COL_RHO]),  "kg/m³")
    eta_c    = pct_control("ec",   "η_c [−]",         to_num(row[COL_ETA_C]), "")

overrides = dict(H0=H0_eff, K=K_eff, rho=rho_eff, eta_a=eta_a, eta_b=eta_b, eta_c=eta_c)

# rango 0 → n_pump_max (50 Hz)
st.caption(f"Rampa del VDF (motor) usada (de 3): {ramp_motor} rpm/s")
traj = simulate(row, 0, int(np_max), ramp_motor, overrides=overrides)

if np.isnan(traj["t_total"]):
    if traj["n_p"].size:
        st.error(f"Cálculo hidráulico no converge: par insuficiente. Se detuvo cerca de n_bomba≈{traj['n_p'][-1]:.0f} rpm, Q≈{traj['Q'][-1]:.0f} m³/h.")
    else:
        st.error("Cálculo hidráulico no converge desde el inicio (par de arranque insuficiente).")
else:
    # comparar con rampa pura
    delta_n_motor_tot = int(np_max) * parts["r"]
    t_rampa_only = delta_n_motor_tot / max(1e-9, ramp_motor)
    t_hid = traj["t_total"]
    limitante = "VDF (rampa)" if t_rampa_only >= t_hid else "Par hidráulico"

    a,b,c = st.columns(3)
    a.markdown(f'<span class="pill">t_hidráulica = {t_hid:.2f} s</span>', unsafe_allow_html=True)
    b.markdown(f'<span class="pill">t_rampa = {t_rampa_only:.2f} s</span>', unsafe_allow_html=True)
    c.markdown(f'<span class="pill">Limitante: {limitante}</span>', unsafe_allow_html=True)

    # gráfico Q(t), n_p(t), P_h(t)
    fig, ax1 = plt.subplots(figsize=(8,3.4), dpi=140)
    t = traj["t"]
    ax1.plot(t, traj["Q"], label="Q (m³/h)")
    ax1.set_xlabel("t (s)"); ax1.set_ylabel("Q (m³/h)")
    ax1.grid(True, alpha=.25)

    ax2 = ax1.twinx()
    ax2.plot(t, traj["n_p"], ls="--", label="n_bomba (rpm)")
    ax2.plot(t, np.array(traj["Ph"])/1000.0, ls=":", label="P_h (kW)")
    ax2.set_ylabel("n_bomba (rpm) / P_h (kW)")

    L1,Lab1 = ax1.get_legend_handles_labels()
    L2,Lab2 = ax2.get_legend_handles_labels()
    ax2.legend(L1+L2, Lab1+Lab2, loc="upper left", fontsize=8)
    st.pyplot(fig, clear_figure=True)

# -------------------- Exportar resumen (todos los TAG) --------------------
st.markdown("<hr/>", unsafe_allow_html=True)
st.subheader("Exportar resumen por TAG (usa la rampa actual)")

def resumen_row(row, ramp_motor):
    J, parts = inercia_eq(row)
    nm_min = to_num(row[COL_NM_MIN]); nm_max = to_num(row[COL_NM_MAX])
    np_max = to_num(row[COL_NP_MAX]); Tnom = to_num(row[COL_T_NOM])
    dn = max(0.0, nm_max - nm_min)
    ndot = n_dot_torque(Tnom, J)
    tpar = dn/max(1e-9, ndot)
    tramp= dn/max(1e-9, ramp_motor)
    tf   = max(tpar, tramp)
    traj = simulate(row, 0, int(np_max), ramp_motor, overrides=None)
    t_h  = traj["t_total"]
    limit = "no converge"
    if not np.isnan(t_h):
        t_r = int(np_max)*parts["r"]/max(1e-9, ramp_motor)
        limit = "VDF (rampa)" if t_r>=t_h else "Par hidráulico"
    return {
        "TAG": str(row[TAG_COL]),
        "J_eq_kgm2": round(J,3),
        "n_dot_torque_rpm_s": round(ndot,2),
        "t_par_sin_s": round(tpar,2),
        "t_rampa_sin_s": round(tramp,2),
        "t_final_sin_s": round(tf,2),
        "t_hid_0_a_npmax_s": (round(t_h,2) if not np.isnan(t_h) else np.nan),
        "limitante_hidraulica": limit
    }

if st.button("Generar CSV"):
    data = [resumen_row(df.iloc[i], ramp_motor) for i in range(len(df))]
    csv = pd.DataFrame(data).to_csv(index=False)
    st.download_button("Descargar resumen.csv", csv, file_name="resumen_por_tag.csv", mime="text/csv")
