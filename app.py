import streamlit as st
import pandas as pd
import numpy as np
from math import pi

st.set_page_config(page_title="Tiempo de reacción – Bombas (VDF)", page_icon="🧮", layout="wide")

# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------
def load_dataset():
    # El archivo está en el repo y se carga siempre desde aquí
    df = pd.read_csv("dataset.csv", sep=";", decimal=",", encoding="utf-8")
    # limpieza mínima de nombres (por si vienen espacios)
    df.columns = [c.strip() for c in df.columns]
    return df

def fmt_num(x, unit="", nd=3):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    try:
        return f"{float(x):,.{nd}f} {unit}".strip()
    except Exception:
        return "—"

# Mapeo de atributos (columna -> metadatos)
ATTRS = {
    "TAG":               {"desc": "Identificador de equipo", "unit": "",            "dtype": "str"},
    "pumpmodel":         {"desc": "Modelo de bomba",         "unit": "",            "dtype": "str"},
    "motorpower_kw":     {"desc": "Potencia motor instalada","unit": "kW",          "dtype": "float"},
    "r_trans":           {"desc": "Relación transmisión r=n_m/n_p", "unit": "—",    "dtype": "float"},
    "motor_n_min_rpm":   {"desc": "Velocidad mínima motor (25 Hz)", "unit": "rpm",  "dtype": "float"},
    "motor_n_max_rpm":   {"desc": "Velocidad máxima motor (50 Hz)", "unit": "rpm",  "dtype": "float"},
    "pump_n_min_rpm":    {"desc": "Velocidad mínima bomba (25 Hz)", "unit": "rpm",  "dtype": "float"},
    "pump_n_max_rpm":    {"desc": "Velocidad máxima bomba (50 Hz)", "unit": "rpm",  "dtype": "float"},
    # Torques e inercias
    "t_nom_nm":                {"desc": "Par motor disponible (constante entre 25–50 Hz)", "unit": "N·m",  "dtype": "float"},
    "motor_j_kgm2":            {"desc": "Inercia motor J_m",                   "unit": "kg·m²", "dtype": "float"},
    "driverpulley_j_kgm2":     {"desc": "Inercia polea motriz J_driver",       "unit": "kg·m²", "dtype": "float"},
    "driverbushing_j_kgm2":    {"desc": "Inercia manguito motriz J_driver_b",  "unit": "kg·m²", "dtype": "float"},
    "drivenpulley_j_Kgm2":     {"desc": "Inercia polea conducida J_driven",    "unit": "kg·m²", "dtype": "float"},
    "drivenbushing_j_Kgm2":    {"desc": "Inercia manguito conducido J_driven_b","unit": "kg·m²","dtype": "float"},
    "impeller_j_kgm2":         {"desc": "Inercia impulsor/rotor bomba J_imp",  "unit": "kg·m²", "dtype": "float"},
}

NUM_INERTIA_COLS = [
    "motor_j_kgm2",
    "driverpulley_j_kgm2",
    "driverbushing_j_kgm2",
    "drivenpulley_j_Kgm2",
    "drivenbushing_j_Kgm2",
    "impeller_j_kgm2",
]

# -----------------------------------------------------------------------------
# Cargar datos
# -----------------------------------------------------------------------------
df = load_dataset()

# Sidebar: selector de TAG
st.sidebar.header("Equipo")
tag = st.sidebar.selectbox("Selecciona TAG", options=df["TAG"].dropna().unique().tolist())

row = df.loc[df["TAG"] == tag].iloc[0]

# -----------------------------------------------------------------------------
# 1) Parámetros
# -----------------------------------------------------------------------------
st.markdown("## 1) Parámetros del equipo")

cols = st.columns(3)
with cols[0]:
    st.markdown("**Identificación**")
    st.write(f"**TAG**: {row['TAG']}")
    st.write(f"**Modelo de bomba**: {row.get('pumpmodel', '—')}")
with cols[1]:
    st.markdown("**Motor & transmisión**")
    st.write(f"**Potencia motor**: {fmt_num(row.get('motorpower_kw'), 'kW', 2)}")
    st.write(f"**Relación r (n_m/n_p)**: {fmt_num(row.get('r_trans'), '', 2)}")
with cols[2]:
    st.markdown("**Rangos de velocidad (25–50 Hz)**")
    st.write(f"**n motor min–max**: {fmt_num(row.get('motor_n_min_rpm'), 'rpm', 0)} – {fmt_num(row.get('motor_n_max_rpm'), 'rpm', 0)}")
    st.write(f"**n bomba  min–max**: {fmt_num(row.get('pump_n_min_rpm'),  'rpm', 0)} – {fmt_num(row.get('pump_n_max_rpm'),  'rpm', 0)}")

st.divider()

# -----------------------------------------------------------------------------
# 2) Cálculo de inercia equivalente
# -----------------------------------------------------------------------------
st.markdown("## 2) Cálculo de inercia equivalente")

c1, c2 = st.columns([1,1])
with c1:
    st.markdown("**Inercias individuales**")
    J_m   = float(row.get("motor_j_kgm2", np.nan))
    J_drv = float(row.get("driverpulley_j_kgm2", np.nan))
    J_db  = float(row.get("driverbushing_j_kgm2", np.nan))
    J_drn = float(row.get("drivenpulley_j_Kgm2", np.nan))
    J_dbb = float(row.get("drivenbushing_j_Kgm2", np.nan))
    J_imp = float(row.get("impeller_j_kgm2", np.nan))
    r     = float(row.get("r_trans", np.nan))

    st.write(f"- **Motor (J_m)**: {fmt_num(J_m, 'kg·m²')}")
    st.write(f"- **Polea motriz (J_driver)**: {fmt_num(J_drv, 'kg·m²')}")
    st.write(f"- **Manguito motriz (J_driver_b)**: {fmt_num(J_db, 'kg·m²')}")
    st.write(f"- **Polea conducida (J_driven)**: {fmt_num(J_drn, 'kg·m²')}")
    st.write(f"- **Manguito conducido (J_driven_b)**: {fmt_num(J_dbb, 'kg·m²')}")
    st.write(f"- **Impulsor/rotor bomba (J_imp)**: {fmt_num(J_imp, 'kg·m²')}")
    st.write(f"- **Relación r (n_m/n_p)**: {fmt_num(r)}")

    # J_eq = J_m + J_driver + J_driver_b + (J_driven + J_driven_b + J_imp)/r^2
    if r and not np.isnan(r):
        J_pump_side = (0.0 if np.isnan(J_drn) else J_drn) \
                    + (0.0 if np.isnan(J_dbb) else J_dbb) \
                    + (0.0 if np.isnan(J_imp) else J_imp)
        J_equiv = (0.0 if np.isnan(J_m) else J_m) \
                + (0.0 if np.isnan(J_drv) else J_drv) \
                + (0.0 if np.isnan(J_db) else J_db) \
                + (J_pump_side / (r**2))
    else:
        J_equiv = np.nan

    st.write(f"- **Inercia equivalente (J_eq)**: {fmt_num(J_equiv, 'kg·m²')}")

with c2:
    st.markdown("**Fórmula utilizada**")
    st.latex(r"J_{eq} \;=\; J_m \;+\; J_{\mathrm{driver}} \;+\; J_{\mathrm{driver\_b}} \;+\; \dfrac{J_{\mathrm{driven}} + J_{\mathrm{driven\_b}} + J_{\mathrm{imp}}}{r^2}")
    st.caption("Las inercias del lado bomba giran a $\\omega_p = \\omega_m/r$. Igualando energías cinéticas a una $\\omega_m$ común se obtiene la división por $r^2$ del término del lado bomba.")
    with st.expander("Formulación de las inercias por componente"):
        st.latex(r"J_{\text{cilindro macizo}}=\tfrac12\,mR^2 \quad\text{(aprox. poleas / manguitos)}")
        st.latex(r"J_{\text{anillo}}=mR^2 \quad\text{(aro delgado)}")
        st.latex(r"J_{\text{conjunto bomba}} \approx J_{\mathrm{driven}}+J_{\mathrm{driven\_b}}+J_{\mathrm{imp}}")
        st.caption("En la app usamos los valores de inercia ya calculados en el dataset para cada pieza.")

st.divider()

# -----------------------------------------------------------------------------
# 3) Tiempo de respuesta (inercial vs. rampa VDF)
# -----------------------------------------------------------------------------
st.markdown("## 3) Tiempo de respuesta (par vs. rampa)")

c1, c2 = st.columns([1,1])

with c1:
    st.markdown("**Supuestos**")
    n1 = float(row.get("motor_n_min_rpm", np.nan))
    n2 = float(row.get("motor_n_max_rpm", np.nan))
    delta_n = (n2 - n1) if (not np.isnan(n1) and not np.isnan(n2)) else np.nan
    T_disp = float(row.get("t_nom_nm", np.nan))  # par disponible constante 25–50 Hz

    st.write(f"- **Rango analizado**: {fmt_num(n1,'rpm',0)} → {fmt_num(n2,'rpm',0)} (Δn = {fmt_num(delta_n,'rpm',0)})")
    st.write(f"- **Par disponible (T_disp)**: {fmt_num(T_disp,'N·m',1)}")
    st.write(f"- **Inercia equivalente (J_eq)**: {fmt_num(J_equiv,'kg·m²')}")

    rampa_vdf = st.slider("Rampa VDF [rpm/s]", min_value=50, max_value=1000, value=300, step=10)

with c2:
    st.markdown("**Cálculo**")
    # n_dot_torque [rpm/s] = (60/(2π)) * (T/J_eq)
    if J_equiv and not np.isnan(J_equiv) and T_disp and not np.isnan(T_disp):
        n_dot_torque = (60.0/(2.0*pi)) * (T_disp / J_equiv)  # rpm/s
        t_par = delta_n / n_dot_torque if delta_n and not np.isnan(delta_n) else np.nan
    else:
        n_dot_torque = np.nan
        t_par = np.nan

    t_rampa = delta_n / rampa_vdf if delta_n and not np.isnan(delta_n) else np.nan

    st.write(f"- **Aceleración por par**: $\\dot n_{{torque}}=\\tfrac{{60}}{{2\\pi}}\\,\\tfrac{{T_{{disp}}}}{{J_{{eq}}}}$ → {fmt_num(n_dot_torque, 'rpm/s', 1)}")
    st.write(f"- **Tiempo por par**: $t_{{par}} = \\Delta n / \\dot n_{{torque}}$ → {fmt_num(t_par, 's', 1)}")
    st.write(f"- **Tiempo por rampa**: $t_{{rampa}} = \\Delta n / \\mathrm{{rampa}}_{{VDF}}$ → {fmt_num(t_rampa, 's', 1)}")

    st.markdown("**Fórmulas**")
    st.latex(r"\dot n_{\mathrm{torque}} = \frac{60}{2\pi}\frac{T_{\mathrm{disp}}}{J_{\mathrm{eq}}}, \qquad t_{\mathrm{par}}=\frac{\Delta n}{\dot n_{\mathrm{torque}}}, \qquad t_{\mathrm{rampa}}=\frac{\Delta n}{\mathrm{rampa}_{\mathrm{VDF}}}")

st.info("Más adelante integraremos el modelo hidráulico ($Q(n), H(Q)$, $\\eta(Q)$) para refinar el tiempo de respuesta con carga.", icon="ℹ️")
