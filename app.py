import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Memoria de Cálculo – Tiempo de reacción (VDF)",
                   page_icon="⏱️", layout="wide")

# ----------------------- Utilidades -----------------------
def to_num(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan

def badge(value, unit="", label="", color="#0a7f45"):
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        txt = "—"
    else:
        txt = f"{value:.2f} {unit}".strip()
    if label:
        label = f"<span style='opacity:.8'>{label}</span> "
    st.markdown(
        f"""
        <div style="
            display:inline-block;
            padding:.30rem .70rem;
            margin:.18rem .28rem;
            border-radius:.60rem;
            background:{color}12;
            border:1px solid {color}55;
            color:{color};
            font-weight:780;">
            {label}{txt}
        </div>
        """,
        unsafe_allow_html=True,
    )

def rpm_to_omega(n_rpm):  # rpm -> rad/s
    return (2.0 * math.pi / 60.0) * n_rpm

def omega_to_rpm(omega):  # rad/s -> rpm
    return (60.0 / (2.0 * math.pi)) * omega

# -------------------- Carga de dataset --------------------
@st.cache_data
def load_dataset():
    path = Path(__file__).parent / "bombas_dataset_with_torque_params.xlsx"
    if not path.exists():
        raise FileNotFoundError("No se encontró 'bombas_dataset_with_torque_params.xlsx' en la raíz del proyecto.")
    try:
        df = pd.read_excel(path, sheet_name="dataSet")
    except Exception:
        df = pd.read_excel(path)

    numeric_cols = [
        "r_trans",
        "motorpower_kw", "t_nom_nm", "motor_j_kgm2", "impeller_j_kgm2",
        "driverpulley_j_kgm2", "driverbushing_j_kgm2", "drivenpulley_j_Kgm2", "drivenbushing_j_Kgm2",
        "motor_n_min_rpm", "motor_n_max_rpm", "pump_n_min_rpm", "pump_n_max_rpm",
        "H0_m", "K_m_s2", "R2_H", "eta_a", "eta_b", "eta_c", "R2_eta",
        "Q_min_m3h", "Q_max_m3h", "Q_ref_m3h", "n_ref_rpm", "rho_kgm3",
        "eta_beta", "eta_min_clip", "eta_max_clip",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = df[c].map(to_num)

    tag_col = df.columns[0]
    df[tag_col] = df[tag_col].astype(str)
    return df, tag_col

try:
    df, TAG_COL = load_dataset()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# ------------------------ Selección ------------------------
st.sidebar.header("Selección")
tags = df[TAG_COL].tolist()
selected = st.sidebar.selectbox("Elige un TAG", tags, index=0)
row = df[df[TAG_COL] == selected].iloc[0]

st.title("Memoria de Cálculo – Tiempo de reacción de bombas (VDF)")

# ------------------ 1) Parámetros entrada -----------------
st.header("1) Parámetros de entrada")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.subheader("Motor")
    badge(to_num(row.get("t_nom_nm")), "Nm", "T_nom")
    badge(to_num(row.get("motor_n_min_rpm")), "rpm", "Velocidad min")
    badge(to_num(row.get("motor_n_max_rpm")), "rpm", "Velocidad max")
    badge(to_num(row.get("motor_j_kgm2")), "kg·m²", "J_m")

with c2:
    st.subheader("Transmisión")
    r = to_num(row.get("r_trans"))
    badge(r, "", "Relación r = n_motor / n_bomba")
    J_driver = (to_num(row.get("driverpulley_j_kgm2")) or 0.0) + (to_num(row.get("driverbushing_j_kgm2")) or 0.0)
    J_driven = (to_num(row.get("drivenpulley_j_Kgm2")) or 0.0) + (to_num(row.get("drivenbushing_j_Kgm2")) or 0.0)
    badge(J_driver, "kg·m²", "J_driver")
    badge(J_driven, "kg·m²", "J_driven")

with c3:
    st.subheader("Bomba")
    badge(to_num(row.get("pump_n_min_rpm")), "rpm", "Velocidad min (≈25 Hz)")
    badge(to_num(row.get("pump_n_max_rpm")), "rpm", "Velocidad max (≈50 Hz)")
    badge(to_num(row.get("impeller_j_kgm2")), "kg·m²", "J_imp (impulsor)")

with c4:
    st.subheader("Sistema (H–Q, η)")
    if {"H0_m","K_m_s2"}.issubset(df.columns):
        st.latex(r"H(Q) = H_0 + K\left(\frac{Q}{3600}\right)^2")
        badge(to_num(row.get("H0_m")), "m", "H0")
        badge(to_num(row.get("K_m_s2")), "m·s²/m⁶", "K")
    if {"eta_a","eta_b","eta_c","Q_ref_m3h"}.issubset(df.columns):
        st.latex(r"\eta(Q) \approx \eta_a + \eta_b \left(\frac{Q}{Q_{\mathrm{ref}}}\right) + \eta_c \left(\frac{Q}{Q_{\mathrm{ref}}}\right)^2")
        badge(to_num(row.get("eta_a")), "", "η_a")
        badge(to_num(row.get("eta_b")), "", "η_b")
        badge(to_num(row.get("eta_c")), "", "η_c")
        badge(to_num(row.get("Q_ref_m3h")), "m³/h", "Q_ref")

# ------------------ 2) Inercia equivalente ----------------
st.header("2) Inercia equivalente al eje del motor")

J_m   = to_num(row.get("motor_j_kgm2")) or 0.0
J_imp = to_num(row.get("impeller_j_kgm2")) or 0.0
J_eq  = np.nan
if r and r > 0:
    J_eq = J_m + J_driver + (J_driven + J_imp) / (r**2)

st.latex(r"J_{\mathrm{eq}} = J_m + J_{\mathrm{driver}} + \dfrac{J_{\mathrm{driven}}+J_{\mathrm{imp}}}{r^2}")
st.caption("Las inercias del lado bomba giran a \( \omega_p=\omega_m/r \). Igualando energías cinéticas a una \( \omega_m \) común se obtiene la división por \( r^2 \).")
st.write("**Sustitución numérica**")
if r and r>0:
    st.latex(rf"J_{{\mathrm{{eq}}}} = {J_m:.2f} + {J_driver:.2f} + \dfrac{{{J_driven:.2f}+{J_imp:.2f}}}{{({r:.2f})^2}}")
badge(J_eq, "kg·m²", "J_eq", color="#0a4")

# ----- función común de tiempos sin hidráulica (por fila) -----
def inertial_times_for_row(row_, rampa_rpmps):
    r_ = to_num(row_.get("r_trans"))
    J_m_ = to_num(row_.get("motor_j_kgm2")) or 0.0
    J_imp_ = to_num(row_.get("impeller_j_kgm2")) or 0.0
    J_driver_ = (to_num(row_.get("driverpulley_j_kgm2")) or 0.0) + (to_num(row_.get("driverbushing_j_kgm2")) or 0.0)
    J_driven_ = (to_num(row_.get("drivenpulley_j_Kgm2")) or 0.0) + (to_num(row_.get("drivenbushing_j_Kgm2")) or 0.0)
    if not r_ or r_ <= 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    J_eq_ = J_m_ + J_driver_ + (J_driven_ + J_imp_) / (r_**2)
    n_ini_ = to_num(row_.get("motor_n_min_rpm"))
    n_fin_ = to_num(row_.get("motor_n_max_rpm"))
    T_nom_ = to_num(row_.get("t_nom_nm"))
    if any(pd.isna([J_eq_, n_ini_, n_fin_, T_nom_])):
        return J_eq_, np.nan, np.nan, np.nan, np.nan
    delta_n_ = max(0.0, n_fin_ - n_ini_)
    n_dot_ = (60.0/(2.0*math.pi)) * (T_nom_ / J_eq_) if J_eq_>0 else np.nan
    t_par_ = (delta_n_ / n_dot_) if n_dot_ and n_dot_>0 else np.nan
    t_rampa_ = (delta_n_ / rampa_rpmps) if rampa_rpmps>0 else np.nan
    t_final_sin_ = np.nanmax([t_par_, t_rampa_])
    return J_eq_, delta_n_, n_dot_, t_par_, t_rampa_, t_final_sin_

# ---------------- 3) Respuesta inercial (sin H) -------------
st.header("3) Respuesta inercial (sin efectos hidráulicos)")

cA, cB, cC = st.columns(3)
n_ini = cA.number_input("Velocidad Motor inicial [rpm]", value=float(to_num(row.get("motor_n_min_rpm") or 500)))
n_fin = cB.number_input("Velocidad Motor final [rpm]",   value=float(to_num(row.get("motor_n_max_rpm") or 1500)))
T_disp = cC.number_input("Par disponible T_nom [Nm]",    value=float(to_num(row.get("t_nom_nm") or 200.0)))

st.latex(r"""\dot n_{\mathrm{torque}}=\frac{60}{2\pi}\frac{T_{\mathrm{disp}}}{J_{\mathrm{eq}}},\qquad 
t_{\mathrm{par}}=\frac{\Delta n}{\dot n_{\mathrm{torque}}},\qquad 
t_{\mathrm{rampa}}=\frac{\Delta n}{\mathrm{rampa}_{\mathrm{VDF}}},\qquad 
t_{\mathrm{final,sin}}=\max(t_{\mathrm{par}},t_{\mathrm{rampa}})""")

delta_n = max(0.0, n_fin - n_ini)
rampa_vdf = st.slider("Rampa del VDF (motor) [rpm/s]", 100, 800, 300, 25)

n_dot_torque = (60.0/(2.0*math.pi)) * (T_disp / J_eq) if (J_eq and J_eq>0) else np.nan
t_par   = (delta_n / n_dot_torque) if n_dot_torque and n_dot_torque>0 else np.nan
t_rampa = (delta_n / rampa_vdf) if rampa_vdf>0 else np.nan
t_final_sin = max(t_par, t_rampa)

cR1, cR2, cR3, cR4 = st.columns(4)
with cR1:
    st.latex(r"\Delta n");                badge(delta_n, "rpm")
with cR2:
    st.latex(r"\dot n_{\mathrm{torque}}"); badge(n_dot_torque, "rpm/s")
with cR3:
    st.latex(r"t_{\mathrm{par}}");        badge(t_par, "s")
with cR4:
    st.latex(r"t_{\mathrm{rampa}}");      badge(t_rampa, "s")

st.markdown(
    f"""<div style="padding:.5rem 1rem;border-radius:.6rem;background:#0a7f4516;border:1px solid #0a7f4555;display:inline-block;">
        <span style="color:#0a7f45;font-weight:800">t_final(sin)</span> = <b>{t_final_sin:.2f} s</b>
    </div>""", unsafe_allow_html=True
)
st.caption("En esta sección no se incluye aún el par hidráulico de la bomba.")

# ----------- helpers hidráulica (datos y simulación) ----------
def has_hyd_data(row_):
    needed = ["H0_m","K_m_s2","Q_min_m3h","Q_max_m3h","Q_ref_m3h",
              "n_ref_rpm","rho_kgm3","eta_a","eta_b","eta_c","eta_min_clip","eta_max_clip"]
    return all(not pd.isna(to_num(row_.get(k))) for k in needed)

def simulate_hydraulics(row_, n_p_ini, n_p_fin, rampa_rpmps):
    """Integra con hidráulica: devuelve t_hid (s) o np.nan si no converge."""
    if not has_hyd_data(row_):
        return np.nan

    r_ = to_num(row_.get("r_trans"))
    if not r_ or r_<=0:
        return np.nan

    # Inercias
    J_m_ = to_num(row_.get("motor_j_kgm2")) or 0.0
    J_imp_ = to_num(row_.get("impeller_j_kgm2")) or 0.0
    J_driver_ = (to_num(row_.get("driverpulley_j_kgm2")) or 0.0) + (to_num(row_.get("driverbushing_j_kgm2")) or 0.0)
    J_driven_ = (to_num(row_.get("drivenpulley_j_Kgm2")) or 0.0) + (to_num(row_.get("drivenbushing_j_Kgm2")) or 0.0)
    J_eq_ = J_m_ + J_driver_ + (J_driven_ + J_imp_) / (r_**2)
    if J_eq_<=0:
        return np.nan

    # Parámetros hidráulicos
    H0   = to_num(row_.get("H0_m"))
    Ksys = to_num(row_.get("K_m_s2"))
    rho  = to_num(row_.get("rho_kgm3"))
    eta_a = to_num(row_.get("eta_a")); eta_b = to_num(row_.get("eta_b")); eta_c = to_num(row_.get("eta_c"))
    eta_min = to_num(row_.get("eta_min_clip")); eta_max = to_num(row_.get("eta_max_clip"))
    Q_ref = to_num(row_.get("Q_ref_m3h")); n_ref = to_num(row_.get("n_ref_rpm"))
    T_disp_ = to_num(row_.get("t_nom_nm"))
    if any(pd.isna([H0,Ksys,rho,eta_a,eta_b,eta_c,eta_min,eta_max,Q_ref,n_ref,T_disp_])):
        return np.nan

    # Dinámica
    direction = 1.0 if n_p_fin >= n_p_ini else -1.0
    n_p = float(n_p_ini)
    omega_m = rpm_to_omega(n_p * r_)
    alpha_vdf_max = (rampa_rpmps * 2.0*math.pi/60.0)  # rad/s²
    dt = 1e-3
    t = 0.0
    max_steps = int(120/dt)
    steps = 0

    while (direction>0 and n_p < n_p_fin) or (direction<0 and n_p > n_p_fin):
        # Caudal (afinidad con n_p)
        Q = Q_ref * (n_p / n_ref)
        Q_min = to_num(row_.get("Q_min_m3h")); Q_max = to_num(row_.get("Q_max_m3h"))
        Q = float(np.clip(Q, Q_min, Q_max))

        # Altura y eficiencia
        H = H0 + Ksys * (Q/3600.0)**2
        eta = eta_a + eta_b*(Q/Q_ref) + eta_c*(Q/Q_ref)**2
        eta = float(np.clip(eta, eta_min, eta_max))
        eta = max(eta, 1e-3)

        # Torque de bomba y reflejado
        omega_p = max(rpm_to_omega(n_p), 1e-3)
        T_pump = (rho * 9.81 * Q * H) / (eta * omega_p)
        T_ref = T_pump / r_

        # Aceleración limitada
        T_net = T_disp_ - T_ref
        if T_net <= 0:
            return np.nan

        alpha_torque = T_net / J_eq_
        alpha = min(alpha_torque, alpha_vdf_max) * direction

        omega_m += alpha * dt
        n_p = omega_to_rpm(omega_m) / r_

        t += dt
        steps += 1
        if steps >= max_steps:
            return np.nan

    return t

# ---------------- 4) Tiempo con hidráulica -----------------
st.header("4) Tiempo de reacción **con** hidráulica")

if not has_hyd_data(row):
    st.info("Para esta sección se requieren H0, K, ρ, η_a, η_b, η_c, Q_ref, n_ref y límites de η. Si están en el Excel, se integrará la dinámica.")
else:
    # Range slider entre 25–50 Hz -> usamos min/max de bomba del dataset
    n_p_min = float(to_num(row.get("pump_n_min_rpm") or 300))
    n_p_max = float(to_num(row.get("pump_n_max_rpm") or 900))
    st.caption(f"Rango permitido por VDF (bomba): **{n_p_min:.0f} – {n_p_max:.0f} rpm** (≈ 25–50 Hz).")
    n_range = st.slider("Rango de velocidad de BOMBA [rpm] (inicio → fin)", 
                        min_value=int(n_p_min), max_value=int(n_p_max),
                        value=(int(n_p_min), int(n_p_max)))
    n_p_ini_sel, n_p_fin_sel = n_range

    st.caption(f"Se usa la **misma rampa** definida en 3): **{rampa_vdf:.0f} rpm/s** en el eje del **motor**.")

    t_hid = simulate_hydraulics(row, n_p_ini_sel, n_p_fin_sel, rampa_vdf)
    if np.isnan(t_hid):
        st.error("No converge con los parámetros actuales (torque insuficiente o límite de tiempo).")
    else:
        st.markdown(
            f"""<div style="padding:.6rem 1rem;border-radius:.6rem;background:#004aad14;border:1px solid #004aad44;display:inline-block;">
                <span style="color:#004aad;font-weight:800">Tiempo de reacción (con hidráulica)</span> = <b>{t_hid:.2f} s</b>
            </div>""", unsafe_allow_html=True
        )

        # Gráficas de referencia H(Q) y η(Q)
        Q_min = to_num(row.get("Q_min_m3h")); Q_max = to_num(row.get("Q_max_m3h"))
        H0 = to_num(row.get("H0_m")); Ksys = to_num(row.get("K_m_s2"))
        eta_a = to_num(row.get("eta_a")); eta_b = to_num(row.get("eta_b")); eta_c = to_num(row.get("eta_c"))
        eta_min = to_num(row.get("eta_min_clip")); eta_max = to_num(row.get("eta_max_clip"))
        Q_ref = to_num(row.get("Q_ref_m3h"))

        Q_grid = np.linspace(Q_min, Q_max, 160)
        H_curve = H0 + Ksys*(Q_grid/3600.0)**2
        eta_curve = eta_a + eta_b*(Q_grid/Q_ref) + eta_c*(Q_grid/Q_ref)**2
        eta_curve = np.clip(eta_curve, eta_min, eta_max)

        g1, g2 = st.columns(2)
        with g1:
            fig, ax = plt.subplots(figsize=(5.6,3.2))
            ax.plot(Q_grid, H_curve, lw=2)
            ax.set_xlabel("Q [m³/h]"); ax.set_ylabel("H [m]")
            ax.set_title("Curva del sistema H(Q)")
            ax.grid(True, alpha=.3)
            st.pyplot(fig)
        with g2:
            fig, ax = plt.subplots(figsize=(5.6,3.2))
            ax.plot(Q_grid, eta_curve*100.0, lw=2, color="#0a7f45")
            ax.set_xlabel("Q [m³/h]"); ax.set_ylabel("η [%]")
            ax.set_title("Eficiencia η(Q)")
            ax.grid(True, alpha=.3)
            st.pyplot(fig)

# ---------------- 5) Resumen y descarga -------------------
st.header("5) Resumen (todos los TAG) y descarga")

def compute_for_all_tags(rampa_rpmps):
    rows = []
    for _, rw in df.iterrows():
        # Inercial por TAG
        J_eq_, delta_n_, n_dot_, t_par_, t_rampa_, t_final_sin_ = inertial_times_for_row(rw, rampa_rpmps)

        # Con hidráulica (si hay datos): usar rango completo 25–50 Hz (min–max del TAG)
        t_hid_ = np.nan
        if has_hyd_data(rw):
            n_p_min_ = to_num(rw.get("pump_n_min_rpm"))
            n_p_max_ = to_num(rw.get("pump_n_max_rpm"))
            if not any(pd.isna([n_p_min_, n_p_max_])):
                t_hid_ = simulate_hydraulics(rw, n_p_min_, n_p_max_, rampa_rpmps)

        rows.append({
            "TAG": str(rw[TAG_COL]),
            "r_trans": to_num(rw.get("r_trans")),
            "T_nom_Nm": to_num(rw.get("t_nom_nm")),
            "J_eq_kgm2": J_eq_,
            "delta_n_rpm": delta_n_,
            "n_dot_torque_rpmps": n_dot_,
            "t_par_s": t_par_,
            "t_rampa_s": t_rampa_,
            "t_final_sin_s": t_final_sin_,
            "t_con_hidraulica_s": t_hid_,
        })
    return pd.DataFrame(rows)

all_df = compute_for_all_tags(rampa_vdf)
st.dataframe(all_df, use_container_width=True, height=360)

csv_all = all_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Descargar reporte (todos los TAG, con rampa seleccionada)",
                   csv_all, file_name=f"reporte_todos_los_TAG_rampa_{int(rampa_vdf)}rpmps.csv",
                   mime="text/csv")

st.markdown("---")
st.caption(
    "Ecuaciones base: \(J_{eq}=J_m+J_{driver}+(J_{driven}+J_{imp})/r^2\), "
    " \( \dot n_{torque}=\frac{60}{2\pi}\frac{T_{disp}}{J_{eq}} \), "
    " \( t_{par}=\Delta n/\dot n_{torque} \), \( t_{rampa}=\Delta n/\mathrm{rampa}_{VDF} \). "
    "Con hidráulica: \( J_{eq}\,\dot\omega_m=T_{disp}-T_{pump}/r \), "
    " \( T_{pump}=\frac{\rho g Q H(Q)}{\eta\,\omega_p} \), \( \omega_p=\omega_m/r \), \( Q\propto n_p \)."
)
