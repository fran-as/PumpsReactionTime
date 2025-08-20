import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Memoria de Cálculo – Tiempo de reacción (VDF)",
                   page_icon="⏱️", layout="wide")

# ---------------------------------------------------------
# Utilidades
# ---------------------------------------------------------
def to_num(x):
    """Convierte valores con coma/punto a float; NaN si no aplica."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan

def get_num(row, name, default=np.nan):
    """Obtiene valor numérico seguro de la fila."""
    if name not in row or pd.isna(row[name]):
        return default
    return to_num(row[name])

def badge(value, unit="", label="", color="#0a7f45"):
    """Chip/etiqueta verde para valores numéricos."""
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
            background:{color}12; /* translúcido */
            border:1px solid {color}55;
            color:{color};
            font-weight:780;
            ">
            {label}{txt}
        </div>
        """,
        unsafe_allow_html=True,
    )

def rpm_to_omega(n_rpm):
    return (2.0 * math.pi / 60.0) * n_rpm

def omega_to_rpm(omega):
    return (60.0 / (2.0 * math.pi)) * omega

# ---------------------------------------------------------
# Carga de datos (desde la raíz del repo)
# ---------------------------------------------------------
@st.cache_data
def load_dataset():
    path = Path(__file__).parent / "bombas_dataset_with_torque_params.xlsx"
    if not path.exists():
        raise FileNotFoundError(
            "No se encontró 'bombas_dataset_with_torque_params.xlsx' en la raíz del proyecto."
        )
    try:
        df = pd.read_excel(path, sheet_name="dataSet")
    except Exception:
        df = pd.read_excel(path)
    # normaliza columnas numéricas (coma/punto)
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
    # asegura primera columna como TAG texto
    tag_col = df.columns[0]
    df[tag_col] = df[tag_col].astype(str)
    return df, tag_col

try:
    df, TAG_COL = load_dataset()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# ---------------------------------------------------------
# Selección de TAG
# ---------------------------------------------------------
st.sidebar.header("Selección")
tags = df[TAG_COL].tolist()
selected = st.sidebar.selectbox("Elige un TAG", tags, index=0)
row = df[df[TAG_COL] == selected].iloc[0]  # <-- SIEMPRE se refresca por TAG

st.title("Memoria de Cálculo – Tiempo de reacción de bombas (VDF)")

# ---------------------------------------------------------
# 1) Parámetros de entrada
# ---------------------------------------------------------
st.header("1) Parámetros de entrada")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.subheader("Motor")
    badge(get_num(row, "t_nom_nm"), "Nm", "T_nom")
    badge(get_num(row, "motor_n_min_rpm"), "rpm", "Velocidad min")
    badge(get_num(row, "motor_n_max_rpm"), "rpm", "Velocidad max")
    badge(get_num(row, "motor_j_kgm2"), "kg·m²", "J_m")

with c2:
    st.subheader("Transmisión")
    r = get_num(row, "r_trans")
    badge(r, "", "Relación r = n_motor / n_bomba")
    J_driver = (get_num(row, "driverpulley_j_kgm2", 0.0) or 0.0) + (get_num(row, "driverbushing_j_kgm2", 0.0) or 0.0)
    J_driven = (get_num(row, "drivenpulley_j_Kgm2", 0.0) or 0.0) + (get_num(row, "drivenbushing_j_Kgm2", 0.0) or 0.0)
    badge(J_driver, "kg·m²", "J_driver (polea+manguito)")
    badge(J_driven, "kg·m²", "J_driven (polea+manguito)")

with c3:
    st.subheader("Bomba")
    badge(get_num(row, "pump_n_min_rpm"), "rpm", "Velocidad min")
    badge(get_num(row, "pump_n_max_rpm"), "rpm", "Velocidad max")
    badge(get_num(row, "impeller_j_kgm2"), "kg·m²", "J_imp (impulsor)")

with c4:
    st.subheader("Sistema (H–Q, η)")
    if {"H0_m","K_m_s2"}.issubset(df.columns):
        st.latex(r"H(Q) = H_0 + K\left(\frac{Q}{3600}\right)^2")
        badge(get_num(row, "H0_m"), "m", "H0")
        badge(get_num(row, "K_m_s2"), "m·s²/m⁶", "K")
    if {"eta_a","eta_b","eta_c","Q_ref_m3h"}.issubset(df.columns):
        st.latex(r"\eta(Q) \approx \eta_a + \eta_b \left(\frac{Q}{Q_{\mathrm{ref}}}\right) + \eta_c \left(\frac{Q}{Q_{\mathrm{ref}}}\right)^2")
        badge(get_num(row, "eta_a"), "", "η_a")
        badge(get_num(row, "eta_b"), "", "η_b")
        badge(get_num(row, "eta_c"), "", "η_c")
        badge(get_num(row, "Q_ref_m3h"), "m³/h", "Q_ref")

# ---------------------------------------------------------
# 2) Inercia equivalente
# ---------------------------------------------------------
st.header("2) Inercia equivalente al eje del motor")

J_m   = get_num(row, "motor_j_kgm2", 0.0) or 0.0
J_imp = get_num(row, "impeller_j_kgm2", 0.0) or 0.0
J_eq  = np.nan
if r and r > 0:
    J_eq = J_m + J_driver + (J_driven + J_imp) / (r**2)

st.latex(r"J_{\mathrm{eq}} = J_m + J_{\mathrm{driver}} + \dfrac{J_{\mathrm{driven}}+J_{\mathrm{imp}}}{r^2}")
st.caption("Las inercias del lado bomba giran a \( \omega_p=\omega_m/r \). Igualando energías cinéticas a una \( \omega_m \) común se obtiene la división por \( r^2 \) del término del lado bomba.")
st.write("**Sustitución numérica**")
if r and r>0:
    st.latex(rf"J_{{\mathrm{{eq}}}} = {J_m:.2f} + {J_driver:.2f} + \dfrac{{{J_driven:.2f}+{J_imp:.2f}}}{{({r:.2f})^2}}")
badge(J_eq, "kg·m²", "J_eq", color="#0a4")

# ---------------------------------------------------------
# 3) Respuesta inercial (sin efectos hidráulicos)
# ---------------------------------------------------------
st.header("3) Respuesta inercial (sin efectos hidráulicos)")

cA, cB, cC = st.columns(3)
n_ini = cA.number_input("Velocidad Motor inicial [rpm]", value=float(get_num(row, "motor_n_min_rpm", 500)))
n_fin = cB.number_input("Velocidad Motor final [rpm]",   value=float(get_num(row, "motor_n_max_rpm", 1500)))
T_disp = cC.number_input("Par disponible T_nom [Nm]",    value=float(get_num(row, "t_nom_nm", 200.0)))

st.latex(r"""\dot n_{\mathrm{torque}}=\frac{60}{2\pi}\frac{T_{\mathrm{disp}}}{J_{\mathrm{eq}}},\qquad 
t_{\mathrm{par}}=\frac{\Delta n}{\dot n_{\mathrm{torque}}},\qquad 
t_{\mathrm{rampa}}=\frac{\Delta n}{\mathrm{rampa}_{\mathrm{VDF}}},\qquad 
t_{\mathrm{final,sin}}=\max(t_{\mathrm{par}},t_{\mathrm{rampa}})""")

delta_n = max(0.0, n_fin - n_ini)
rampa_vdf = st.slider("Rampa del VDF (rpm/s)", 100, 800, 300, 25)

n_dot_torque = (60.0/(2.0*math.pi)) * (T_disp / J_eq) if (J_eq and J_eq>0) else np.nan
t_par   = (delta_n / n_dot_torque) if n_dot_torque and n_dot_torque>0 else np.nan
t_rampa = (delta_n / rampa_vdf) if rampa_vdf>0 else np.nan
t_final_sin = max(t_par, t_rampa)

# Resultados en fila (mismo renglón)
cR1, cR2, cR3, cR4 = st.columns(4)
with cR1:
    st.latex(r"\Delta n")
    badge(delta_n, "rpm", "", "#0a7f45")
with cR2:
    st.latex(r"\dot n_{\mathrm{torque}}")
    badge(n_dot_torque, "rpm/s", "", "#0a7f45")
with cR3:
    st.latex(r"t_{\mathrm{par}}")
    badge(t_par, "s", "", "#0a7f45")
with cR4:
    st.latex(r"t_{\mathrm{rampa}}")
    badge(t_rampa, "s", "", "#0a7f45")

st.markdown(
    f"""<div style="padding:.5rem 1rem;border-radius:.6rem;background:#0a7f4516;border:1px solid #0a7f4555;display:inline-block;">
        <span style="color:#0a7f45;font-weight:800">t_final(sin)</span> = <b>{t_final_sin:.2f} s</b>
    </div>""", unsafe_allow_html=True
)
st.caption("Aquí no se incluye el par hidráulico de la bomba.")

# ---------------------------------------------------------
# 4) Tiempo de reacción CON hidráulica (integración)
# ---------------------------------------------------------
st.header("4) Tiempo de reacción **con** hidráulica")

needed = {"H0_m","K_m_s2","Q_min_m3h","Q_max_m3h","Q_ref_m3h","n_ref_rpm",
          "rho_kgm3","eta_a","eta_b","eta_c","eta_min_clip","eta_max_clip"}
if not needed.issubset(df.columns) or any(pd.isna(get_num(row, c)) for c in needed):
    st.info("Para esta sección se requieren H0, K, ρ, η_a, η_b, η_c, Q_ref, n_ref y límites de η. Si están en el Excel, se integrará la dinámica.")
else:
    # Parámetros hidráulicos
    H0   = get_num(row, "H0_m")
    Ksys = get_num(row, "K_m_s2")
    rho  = get_num(row, "rho_kgm3")
    eta_a = get_num(row, "eta_a"); eta_b = get_num(row, "eta_b"); eta_c = get_num(row, "eta_c")
    eta_min = get_num(row, "eta_min_clip"); eta_max = get_num(row, "eta_max_clip")
    Q_ref = get_num(row, "Q_ref_m3h"); n_ref = get_num(row, "n_ref_rpm")
    Q_min = get_num(row, "Q_min_m3h"); Q_max = get_num(row, "Q_max_m3h")

    # Curvas H(Q) y η(Q) para referenciar
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

    st.subheader("Rango de evaluación (bomba)")
    cP1, cP2, cP3 = st.columns(3)
    n_p_ini = cP1.number_input("n_bomba inicial [rpm]",
                               value=float(get_num(row, "pump_n_min_rpm", 300)))
    n_p_fin = cP2.number_input("n_bomba final [rpm]",
                               value=float(get_num(row, "pump_n_max_rpm", 900)))
    rampa_vdf2 = cP3.slider("Rampa VDF (motor) [rpm/s]", 100, 800, rampa_vdf, 25,
                            help="Máxima aceleración impuesta por el VDF (en el eje del motor).")

    # Integración simple con paso fijo
    # J_eq dω_m/dt = T_disp - T_pump/r
    # T_pump = ρ g Q H / (η ω_p)
    if any(np.isnan([J_eq, r, T_disp])) or r<=0 or J_eq<=0:
        st.warning("Faltan parámetros de inercia/relación/torque para integrar.")
    else:
        # límites de seguridad
        n_p_ini, n_p_fin = float(n_p_ini), float(n_p_fin)
        direction = 1.0 if n_p_fin >= n_p_ini else -1.0
        n_p = n_p_ini
        omega_m = rpm_to_omega(n_p * r)  # ω_m = r * ω_p
        omega_p = omega_m / r

        alpha_vdf_max = (rampa_vdf2 * 2.0*math.pi/60.0)  # rad/s²
        dt = 1e-3  # s
        t = 0.0
        n_hist, q_hist, t_hist = [], [], []

        success = True
        max_steps = int(120/dt)  # tope duro 120 s
        steps = 0
        while (direction>0 and n_p < n_p_fin) or (direction<0 and n_p > n_p_fin):
            # Caudal por afinidad
            Q = Q_ref * (n_p / n_ref)
            Q = float(np.clip(Q, Q_min, Q_max))

            # Altura y eficiencia
            H = H0 + Ksys * (Q/3600.0)**2
            eta = eta_a + eta_b*(Q/Q_ref) + eta_c*(Q/Q_ref)**2
            eta = float(np.clip(eta, eta_min, eta_max))
            eta = max(eta, 1e-3)  # evita dividir por ~0

            # Torque en el eje BOMBA
            omega_p = max(rpm_to_omega(n_p), 1e-3)
            T_pump = (rho * 9.81 * Q * H) / (eta * omega_p)  # Nm

            # Torque reflejado al motor
            T_ref = T_pump / r

            # Aceleración limitada por VDF y por torque
            T_net = T_disp - T_ref
            if T_net <= 0:
                success = False
                break

            alpha_torque = T_net / J_eq  # rad/s² en el motor
            alpha = min(alpha_torque, alpha_vdf_max) * direction

            # Paso de integración (Euler)
            omega_m += alpha * dt
            n_p = omega_to_rpm(omega_m) / r
            t += dt
            steps += 1
            n_hist.append(n_p); q_hist.append(Q); t_hist.append(t)

            if steps >= max_steps:
                success = False
                break

        if not success:
            st.error("El torque disponible no alcanza (o se alcanzó el tope de tiempo). Revisa T_nom, r, J_eq, η(Q) o el rango elegido.")
        else:
            t_hid = t
            st.markdown(
                f"""<div style="padding:.6rem 1rem;border-radius:.6rem;background:#004aad14;border:1px solid #004aad44;display:inline-block;">
                    <span style="color:#004aad;font-weight:800">Tiempo de reacción (con hidráulica)</span> = <b>{t_hid:.2f} s</b>
                </div>""", unsafe_allow_html=True
            )

            # Gráficos n_p(t) y Q(t)
            fig, ax = plt.subplots(1,2, figsize=(11.5,3.6))
            ax[0].plot(t_hist, n_hist, lw=2)
            ax[0].set_title("Velocidad de bomba vs tiempo")
            ax[0].set_xlabel("t [s]"); ax[0].set_ylabel("n_bomba [rpm]")
            ax[0].grid(True, alpha=.3)

            ax[1].plot(t_hist, q_hist, lw=2, color="#0a7f45")
            ax[1].set_title("Caudal vs tiempo")
            ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("Q [m³/h]")
            ax[1].grid(True, alpha=.3)

            st.pyplot(fig)

# ---------------------------------------------------------
# 5) Resumen y descarga
# ---------------------------------------------------------
st.header("5) Resumen y descarga")

summary = {
    "TAG": selected,
    "r_trans": r,
    "T_nom_Nm": get_num(row, "t_nom_nm"),
    "n_motor_min_rpm": get_num(row, "motor_n_min_rpm"),
    "n_motor_max_rpm": get_num(row, "motor_n_max_rpm"),
    "n_pump_min_rpm": get_num(row, "pump_n_min_rpm"),
    "n_pump_max_rpm": get_num(row, "pump_n_max_rpm"),
    "J_m_kgm2": J_m,
    "J_driver_kgm2": J_driver,
    "J_driven_kgm2": J_driven,
    "J_imp_kgm2": J_imp,
    "J_eq_kgm2": J_eq,
    "rampa_VDF_rpmps": rampa_vdf,
    "delta_n_rpm": delta_n,
    "n_dot_torque_rpmps": n_dot_torque,
    "t_par_s": t_par,
    "t_rampa_s": t_rampa,
    "t_final_sin_s": t_final_sin,
}
out_df = pd.DataFrame([summary])
st.dataframe(out_df, use_container_width=True)

csv = out_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Descargar resumen CSV", csv,
                   file_name=f"reaccion_{selected}.csv", mime="text/csv")

st.markdown("---")
st.caption(
    "Ecuaciones: \(J_{eq}=J_m+J_{driver}+(J_{driven}+J_{imp})/r^2\), "
    " \( \dot n_{torque}=\frac{60}{2\pi}\frac{T_{disp}}{J_{eq}} \), "
    " \( t_{par}=\Delta n/\dot n_{torque} \), "
    " \( t_{rampa}=\Delta n/\mathrm{rampa}_{VDF} \). "
    "Con hidráulica: \( J_{eq}\,\dot\omega_m=T_{disp}-T_{pump}/r \), "
    " \( T_{pump}=\frac{\rho g Q H(Q)}{\eta\,\omega_p} \), \( \omega_p=\omega_m/r \), \( Q\propto n_p \)."
)
