import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Memoria de Cálculo – Tiempo de reacción (VDF)",
                   page_icon="⏱️", layout="wide")

# ---------------------------
# Utilidades
# ---------------------------

def to_num(x):
    """Convierte a número admitiendo coma decimal; deja NaN si no aplica."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    # reemplaza coma por punto; quita espacios
    s = s.replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan

def badge(value, unit="", label="", color="#0a7f45"):
    txt = f"{value:.2f} {unit}".strip()
    if label:
        label = f"<span style='opacity:.8'>{label}</span> "
    return st.markdown(
        f"""
        <div style="
            display:inline-block;
            padding:.25rem .65rem;
            margin:.15rem .25rem;
            border-radius:.50rem;
            background:{color}1A; /* translúcido */
            border:1px solid {color}55;
            color:{color};
            font-weight:700;
            ">
            {label}{txt}
        </div>
        """, unsafe_allow_html=True
    )

def col(df, name):
    """Devuelve la serie de la columna exacta (nombre ya dado). 
       Aplica conversión numérica si corresponde."""
    s = df[name] if name in df.columns else pd.Series([np.nan]*len(df))
    # Si luce numérica, conviértela
    if s.dtype == object:
        return s.map(to_num)
    return pd.to_numeric(s, errors="coerce")

# ---------------------------
# Carga de datos
# ---------------------------

st.sidebar.header("Datos")
file = st.sidebar.file_uploader(
    "Sube `bombas_dataset_with_torque_params.xlsx`",
    type=["xlsx"],
    help="Debe contener una hoja con la tabla única 'dataSet' o una tabla con los encabezados indicados."
)

if file is None:
    st.info("Sube el archivo Excel para comenzar.")
    st.stop()

# intenta leer una tabla llamada dataSet; si no, usa la primera hoja
try:
    df = pd.read_excel(file, sheet_name="dataSet")
except Exception:
    df = pd.read_excel(file)

# Garantiza que el primer campo sea el TAG (primera columna)
tag_col = df.columns[0]
df[tag_col] = df[tag_col].astype(str)

# Asegura columnas numéricas según nombres indicados
num_cols = [
    "r_trans",
    "motorpower_kw", "t_nom_nm", "motor_j_kgm2", "impeller_j_kgm2",
    "driverpulley_j_kgm2", "driverbushing_j_kgm2", "drivenpulley_j_Kgm2", "drivenbushing_j_Kgm2",
    "motor_n_min_rpm", "motor_n_max_rpm", "pump_n_min_rpm", "pump_n_max_rpm",
    "H0_m", "K_m_s2", "R2_H", "eta_a", "eta_b", "eta_c", "R2_eta",
    "Q_min_m3h", "Q_max_m3h", "Q_ref_m3h", "n_ref_rpm", "rho_kgm3",
    "eta_beta", "eta_min_clip", "eta_max_clip"
]
for c in num_cols:
    if c in df.columns:
        df[c] = df[c].map(to_num)

# ---------------------------
# Selección de TAG
# ---------------------------

tags = df[tag_col].tolist()
st.sidebar.subheader("Selección")
selected = st.sidebar.selectbox("Elige un TAG", tags, index=0)
row = df[df[tag_col] == selected].iloc[0]

st.title("Memoria de Cálculo – Tiempo de reacción de bombas (VDF)")

# ---------------------------
# 1) Parámetros de entrada
# ---------------------------

st.header("1) Parámetros de entrada")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.subheader("Motor")
    badge(row["t_nom_nm"], "Nm", "T_nom")
    badge(row["motor_n_min_rpm"], "rpm", "Velocidad min")
    badge(row["motor_n_max_rpm"], "rpm", "Velocidad max")
    badge(row["motor_j_kgm2"], "kg·m²", "J_m (inercia)")

with col2:
    st.subheader("Transmisión")
    badge(row["r_trans"], "", "Relación r = n_motor / n_bomba")
    J_driver = (row.get("driverpulley_j_kgm2", 0.0) or 0.0) + (row.get("driverbushing_j_kgm2", 0.0) or 0.0)
    J_driven = (row.get("drivenpulley_j_Kgm2", 0.0) or 0.0) + (row.get("drivenbushing_j_Kgm2", 0.0) or 0.0)
    badge(J_driver, "kg·m²", "J_driver (polea+manguito)")
    badge(J_driven, "kg·m²", "J_driven (polea+manguito)")

with col3:
    st.subheader("Bomba")
    badge(row["pump_n_min_rpm"], "rpm", "Velocidad min")
    badge(row["pump_n_max_rpm"], "rpm", "Velocidad max")
    badge(row["impeller_j_kgm2"], "kg·m²", "J_imp (impulsor)")

with col4:
    st.subheader("Sistema (H–Q, η)")
    if {"H0_m","K_m_s2"}.issubset(df.columns):
        st.latex(r"H(Q) = H_0 + K \left(\tfrac{Q}{3600}\right)^2")
        badge(row["H0_m"], "m", "H0")
        badge(row["K_m_s2"], "m·s²/m⁶", "K")
    if {"eta_a","eta_b","eta_c"}.issubset(df.columns):
        st.latex(r"\eta(Q) \approx \eta_a + \eta_b \left(\tfrac{Q}{Q_{\mathrm{ref}}}\right) + \eta_c \left(\tfrac{Q}{Q_{\mathrm{ref}}}\right)^2")
        badge(row.get("eta_a", np.nan), "", "η_a")
        badge(row.get("eta_b", np.nan), "", "η_b")
        badge(row.get("eta_c", np.nan), "", "η_c")

# ---------------------------
# 2) Inercia equivalente al eje del motor
# ---------------------------

st.header("2) Inercia equivalente al eje del motor")

r = row["r_trans"] if not pd.isna(row["r_trans"]) else np.nan
J_m = row["motor_j_kgm2"] if not pd.isna(row["motor_j_kgm2"]) else 0.0
J_imp = row["impeller_j_kgm2"] if not pd.isna(row["impeller_j_kgm2"]) else 0.0

# Nota: las inercias del LADO BOMBA deben reflejarse al eje motor dividiendo por r^2
J_eq = J_m + J_driver + (J_driven + J_imp) / (r**2) if r and r>0 else np.nan

st.latex(r"J_{\mathrm{eq}} \;=\; J_m \;+\; J_{\mathrm{driver}} \;+\; \dfrac{J_{\mathrm{driven}} + J_{\mathrm{imp}}}{r^2}")
st.caption("Las inercias del lado bomba giran a ω_p = ω_m / r. Igualando energías cinéticas a una ω_m común se obtiene la división por r² del término del lado bomba.")

st.write("**Sustitución numérica**")
st.latex(
    rf"J_{{\mathrm{{eq}}}} = {J_m:.2f} + {J_driver:.2f} + \dfrac{{{J_driven:.2f}+{J_imp:.2f}}}{{({r:.2f})^2}}"
)
badge(J_eq, "kg·m²", "J_eq", color="#0a4")

# ---------------------------
# 3) Respuesta inercial (sin efectos hidráulicos)
# ---------------------------

st.header("3) Respuesta inercial (sin efectos hidráulicos)")

cA, cB, cC = st.columns(3)

with cA:
    n_ini = float(st.number_input("Velocidad Motor inicial [rpm]", value=float(row["motor_n_min_rpm"])))
with cB:
    n_fin = float(st.number_input("Velocidad Motor final [rpm]", value=float(row["motor_n_max_rpm"])))
with cC:
    T_disp = float(st.number_input("Par disponible T_nom [Nm]", value=float(row["t_nom_nm"])))

st.latex(r"""\dot n_{\mathrm{torque}}=\frac{60}{2\pi}\frac{T_{\mathrm{disp}}}{J_{\mathrm{eq}}},\qquad 
t_{\mathrm{par}}=\frac{\Delta n}{\dot n_{\mathrm{torque}}},\qquad 
t_{\mathrm{rampa}}=\frac{\Delta n}{\mathrm{rampa}_{\mathrm{VDF}}},\qquad 
t_{\mathrm{final,sin}}=\max(t_{\mathrm{par}},t_{\mathrm{rampa}})""")

delta_n = max(0.0, n_fin - n_ini)
rampa_vdf = st.slider("Rampa del VDF (rpm/s)", min_value=100, max_value=800, value=300, step=25)

n_dot_torque = (60.0/(2.0*math.pi)) * (T_disp / J_eq) if (J_eq and J_eq>0) else np.nan
t_par = (delta_n / n_dot_torque) if n_dot_torque and n_dot_torque>0 else np.nan
t_rampa = (delta_n / rampa_vdf) if rampa_vdf>0 else np.nan
t_final_sin = max(t_par, t_rampa)

badge(delta_n, "rpm", r"\Delta n", "#095")
badge(n_dot_torque, "rpm/s", r"\dot n_{\mathrm{torque}}", "#095")
badge(t_par, "s", r"t_{\mathrm{par}}", "#095")
badge(t_rampa, "s", r"t_{\mathrm{rampa}}", "#095")
badge(t_final_sin, "s", r"t_{\mathrm{final,sin}}", "#0a7f45")

st.caption("En esta sección aún no se incluye el par hidráulico de la bomba; solo la inercia mecánica y la rampa del VDF.")

# ---------------------------
# 4) (Opcional) Respuesta con hidráulica: visual (si existen parámetros)
# ---------------------------

st.header("4) Respuesta con hidráulica (visual)")

have_system = {"H0_m","K_m_s2","Q_min_m3h","Q_max_m3h","Q_ref_m3h","n_ref_rpm","rho_kgm3","eta_a","eta_b","eta_c","eta_min_clip","eta_max_clip"}.issubset(df.columns)
if not have_system or any(pd.isna(row.get(c, np.nan)) for c in ["H0_m","K_m_s2","Q_min_m3h","Q_max_m3h","Q_ref_m3h","n_ref_rpm","rho_kgm3","eta_a","eta_b","eta_c","eta_min_clip","eta_max_clip"]):
    st.info("Para esta vista se requieren parámetros del sistema (H0, K, ρ, η_a, η_b, η_c, Q_ref, límites de η, etc.). Si están en el Excel, la app los utilizará automáticamente.")
else:
    # Curvas H(Q) y eficiencia(Q)
    Q_min = float(row["Q_min_m3h"])
    Q_max = float(row["Q_max_m3h"])
    Q_ref = float(row["Q_ref_m3h"])

    Q_grid = np.linspace(Q_min, Q_max, 120)
    H0 = float(row["H0_m"])
    K = float(row["K_m_s2"])
    H = H0 + K * (Q_grid/3600.0)**2

    eta_a = float(row["eta_a"]); eta_b = float(row["eta_b"]); eta_c = float(row["eta_c"])
    eta = eta_a + eta_b*(Q_grid/Q_ref) + eta_c*(Q_grid/Q_ref)**2
    eta = np.clip(eta, float(row["eta_min_clip"]), float(row["eta_max_clip"]))

    fig, ax = plt.subplots(1,2, figsize=(11,3.8))
    ax[0].plot(Q_grid, H, lw=2)
    ax[0].set_title("Curva del sistema H(Q)")
    ax[0].set_xlabel("Q [m³/h]"); ax[0].set_ylabel("H [m]")
    ax[0].grid(True, alpha=.3)

    ax[1].plot(Q_grid, eta*100, lw=2, color="#0a7f45")
    ax[1].set_title("Eficiencia estimada η(Q)")
    ax[1].set_xlabel("Q [m³/h]"); ax[1].set_ylabel("η [%]")
    ax[1].grid(True, alpha=.3)
    st.pyplot(fig)

    st.caption("Estas curvas se usan para enriquecer el modelo dinámico (T_load = ρ g Q H / ω). Si lo habilitas, la integración resuelve J_eq dω/dt = T_disp − T_load/r.")

# ---------------------------
# 5) Exportar resumen del TAG
# ---------------------------

st.header("5) Resumen y exportación")

summary = {
    "TAG": selected,
    "r_trans": r,
    "T_nom_Nm": row["t_nom_nm"],
    "n_motor_min_rpm": row["motor_n_min_rpm"],
    "n_motor_max_rpm": row["motor_n_max_rpm"],
    "n_pump_min_rpm": row["pump_n_min_rpm"],
    "n_pump_max_rpm": row["pump_n_max_rpm"],
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
    "t_final_sin_s": t_final_sin
}
out_df = pd.DataFrame([summary])
st.dataframe(out_df, use_container_width=True)

csv = out_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Descargar resumen CSV", csv, file_name=f"reaccion_{selected}.csv", mime="text/csv")

st.markdown("---")
st.caption(
    "Ecuaciones base: \( J_{\mathrm{eq}} = J_m + J_{\mathrm{driver}} + \frac{J_{\mathrm{driven}}+J_{\mathrm{imp}}}{r^2} \), "
    " \( \dot n_{\mathrm{torque}}=\frac{60}{2\pi}\frac{T_{\mathrm{disp}}}{J_{\mathrm{eq}}} \), "
    " \( t_{\mathrm{par}}=\frac{\Delta n}{\dot n_{\mathrm{torque}}} \), "
    " \( t_{\mathrm{rampa}}=\frac{\Delta n}{\mathrm{rampa}_{\mathrm{VDF}}} \).  "
    "Para integrar hidráulica: \( J_{\mathrm{eq}}\dot\omega_m = T_{\mathrm{disp}} - \frac{T_{\mathrm{pump}}}{r} \) con "
    " \( T_{\mathrm{pump}} = \frac{\rho g Q H(Q)}{\omega_p} \) y \( \omega_p=\omega_m/r \)."
)
