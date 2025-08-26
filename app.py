import streamlit as st
import pandas as pd
import numpy as np
from math import pi

st.set_page_config(page_title="Tiempo de reacción – Bombas (VDF)", page_icon="🧮", layout="wide")

# =============================================================================
# Utilidades
# =============================================================================
PARAM_COLOR = "#1f6feb"   # azul (datos dados)
CALC_COLOR  = "#1a7f37"   # verde (valores calculados)

def colored_value(text: str, kind: str = "param") -> str:
    color = PARAM_COLOR if kind == "param" else CALC_COLOR
    return f"<span style='color:{color}; font-weight:600'>{text}</span>"

def fmt_num(x, unit: str = "", nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    try:
        s = f"{float(x):,.{nd}f}"
        if unit:
            s += f" {unit}"
        return s
    except Exception:
        return "—"

def load_dataset() -> pd.DataFrame:
    # Se asume dataset.csv siempre presente en el repo
    df = pd.read_csv("dataset.csv", sep=";", decimal=",", encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    return df

# Mapeo explícito de columnas usadas
ATTRS = {
    "TAG":                   {"desc": "Identificador de equipo"},
    "pumpmodel":             {"desc": "Modelo de bomba"},
    "impeller_d_mm":         {"desc": "Diámetro impulsor"},
    "motorpower_kw":         {"desc": "Potencia motor instalada"},
    "r_trans":               {"desc": "Relación transmisión r = n_m/n_p"},
    "t_nom_nm":              {"desc": "Par motor disponible (25–50 Hz)"},
    "motor_n_min_rpm":       {"desc": "Velocidad motor mínima (25 Hz)"},
    "motor_n_max_rpm":       {"desc": "Velocidad motor máxima (50 Hz)"},
    "pump_n_min_rpm":        {"desc": "Velocidad bomba mínima (25 Hz)"},
    "pump_n_max_rpm":        {"desc": "Velocidad bomba máxima (50 Hz)"},
    "motor_j_kgm2":          {"desc": "Inercia motor J_m"},
    "driverpulley_j_kgm2":   {"desc": "Inercia polea motriz J_driver"},
    "driverbushing_j_kgm2":  {"desc": "Inercia manguito motriz J_driver_b"},
    "drivenpulley_j_Kgm2":   {"desc": "Inercia polea conducida J_driven"},
    "drivenbushing_j_Kgm2":  {"desc": "Inercia manguito conducido J_driven_b"},
    "impeller_j_kgm2":       {"desc": "Inercia impulsor/rotor J_imp"},
}

# =============================================================================
# Carga de datos
# =============================================================================
df = load_dataset()

# Sidebar
st.sidebar.header("Equipo")
tag = st.sidebar.selectbox("Selecciona TAG", options=df["TAG"].dropna().unique().tolist())
row = df.loc[df["TAG"] == tag].iloc[0]

# =============================================================================
# 1) Parámetros del equipo
# =============================================================================
st.markdown("## 1) Parámetros del equipo")

c_id, c_mt, c_rng = st.columns(3)

with c_id:
    st.markdown("**Identificación**")
    st.markdown(f"**TAG**: {colored_value(str(row.get('TAG', '—')), 'param')}", unsafe_allow_html=True)
    st.markdown(f"**Modelo de bomba**: {colored_value(str(row.get('pumpmodel', '—')), 'param')}", unsafe_allow_html=True)
    st.markdown(
        f"**Diámetro de impulsor**: {colored_value(fmt_num(row.get('impeller_d_mm'), 'mm', 0), 'param')}",
        unsafe_allow_html=True,
    )

with c_mt:
    st.markdown("**Motor & transmisión**")
    st.markdown(
        f"**Potencia motor**: {colored_value(fmt_num(row.get('motorpower_kw'), 'kW', 2), 'param')}",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"**Relación r (n_m/n_p)**: {colored_value(fmt_num(row.get('r_trans'), '', 2), 'param')}",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"**Torque nominal (T_nom)**: {colored_value(fmt_num(row.get('t_nom_nm'), 'N·m', 1), 'param')}",
        unsafe_allow_html=True,
    )

with c_rng:
    st.markdown("**Rangos de velocidad (25–50 Hz)**")
    n1m = row.get("motor_n_min_rpm")
    n2m = row.get("motor_n_max_rpm")
    n1p = row.get("pump_n_min_rpm")
    n2p = row.get("pump_n_max_rpm")

    mm_html = f"{colored_value(fmt_num(n1m, 'rpm', 0), 'param')} – {colored_value(fmt_num(n2m, 'rpm', 0), 'param')}"
    bp_html = f"{colored_value(fmt_num(n1p, 'rpm', 0), 'param')} – {colored_value(fmt_num(n2p, 'rpm', 0), 'param')}"
    st.markdown(f"**Velocidad motor min–max**: {mm_html}", unsafe_allow_html=True)
    st.markdown(f"**Velocidad bomba min–max**: {bp_html}", unsafe_allow_html=True)

st.divider()

# =============================================================================
# 2) Cálculo de inercia equivalente
# =============================================================================
st.markdown("## 2) Cálculo de inercia equivalente")

c_l, c_r = st.columns([1, 1])

# Lectura de inercias y relación
J_m   = float(row.get("motor_j_kgm2", np.nan))
J_drv = float(row.get("driverpulley_j_kgm2", np.nan))
J_db  = float(row.get("driverbushing_j_kgm2", np.nan))
J_drn = float(row.get("drivenpulley_j_Kgm2", np.nan))
J_dbb = float(row.get("drivenbushing_j_Kgm2", np.nan))
J_imp = float(row.get("impeller_j_kgm2", np.nan))
r     = float(row.get("r_trans", np.nan))

# Cálculo de J_eq
if r and not np.isnan(r):
    J_pump_side = (0.0 if np.isnan(J_drn) else J_drn) \
                + (0.0 if np.isnan(J_dbb) else J_dbb) \
                + (0.0 if np.isnan(J_imp) else J_imp)
    J_eq = (0.0 if np.isnan(J_m) else J_m) \
         + (0.0 if np.isnan(J_drv) else J_drv) \
         + (0.0 if np.isnan(J_db) else J_db) \
         + (J_pump_side / (r**2))
else:
    J_eq = np.nan

with c_l:
    st.markdown("**Inercias individuales**")
    st.markdown(f"- **Motor (J_m)**: {colored_value(fmt_num(J_m, 'kg·m²'), 'param')}", unsafe_allow_html=True)
    st.markdown(f"- **Polea motriz (J_driver)**: {colored_value(fmt_num(J_drv, 'kg·m²'), 'param')}", unsafe_allow_html=True)
    st.markdown(f"- **Manguito motriz (J_driver_b)**: {colored_value(fmt_num(J_db, 'kg·m²'), 'param')}", unsafe_allow_html=True)
    st.markdown(f"- **Polea conducida (J_driven)**: {colored_value(fmt_num(J_drn, 'kg·m²'), 'param')}", unsafe_allow_html=True)
    st.markdown(f"- **Manguito conducido (J_driven_b)**: {colored_value(fmt_num(J_dbb, 'kg·m²'), 'param')}", unsafe_allow_html=True)
    st.markdown(f"- **Impulsor/rotor bomba (J_imp)**: {colored_value(fmt_num(J_imp, 'kg·m²'), 'param')}", unsafe_allow_html=True)
    st.markdown(f"- **Relación r (n_m/n_p)**: {colored_value(fmt_num(r, '', 2), 'param')}", unsafe_allow_html=True)
    st.markdown(f"- **Inercia equivalente (J_eq)**: {colored_value(fmt_num(J_eq, 'kg·m²'), 'calc')}", unsafe_allow_html=True)

with c_r:
    st.markdown("**Fórmula utilizada**")
    st.latex(r"J_{eq} = J_m + J_{\mathrm{driver}} + J_{\mathrm{driver\_b}} + \dfrac{J_{\mathrm{driven}} + J_{\mathrm{driven\_b}} + J_{\mathrm{imp}}}{r^2}")
    st.caption("Las inercias del lado bomba giran a $\\omega_p = \\omega_m/r$. Igualando energías cinéticas a una $\\omega_m$ común se obtiene la división por $r^2$ para términos del lado bomba.")
    with st.expander("Formulación de las inercias por componente"):
        st.latex(r"J_{\text{cilindro macizo}}=\tfrac12\,mR^2 \quad\text{(aprox. poleas/manguitos)}")
        st.latex(r"J_{\text{anillo}}=mR^2 \quad\text{(aro delgado)}")
        st.markdown(
            "- **Motor (J_m):** valores tomados de **hoja de datos WEG**.\n"
            "- **Poleas (J_driver, J_driven):** valores tomados de **catálogo TB Woods**.\n"
            "- **Manguitos (J_driver_b, J_driven_b):** **aproximados como el 10%** de la inercia de su polea correspondiente.\n"
            "- **Impulsor (J_imp):** valores tomados de **manuales Metso**.",
        )

st.divider()

# =============================================================================
# 3) Tiempo de respuesta (par vs. rampa)
# =============================================================================
st.markdown("## 3) Tiempo de respuesta (par vs. rampa)")

c_a, c_b = st.columns([1, 1])

n_min = float(row.get("motor_n_min_rpm", np.nan))
n_max = float(row.get("motor_n_max_rpm", np.nan))
Delta_n = (n_max - n_min) if (not np.isnan(n_min) and not np.isnan(n_max)) else np.nan
T_disp = float(row.get("t_nom_nm", np.nan))  # par disponible (25–50 Hz)

with c_a:
    st.markdown("**Supuestos y entradas**")
    st.markdown(f"- **Rango de análisis**: {colored_value(fmt_num(n_min, 'rpm', 0), 'param')} → {colored_value(fmt_num(n_max, 'rpm', 0), 'param')}", unsafe_allow_html=True)
    st.markdown(f"- **Δn**: {colored_value(fmt_num(Delta_n, 'rpm', 0), 'calc')}", unsafe_allow_html=True)
    st.markdown(f"- **Par disponible (T_disp)**: {colored_value(fmt_num(T_disp, 'N·m', 1), 'param')}", unsafe_allow_html=True)
    st.markdown(f"- **Inercia equivalente (J_eq)**: {colored_value(fmt_num(J_eq, 'kg·m²'), 'calc')}", unsafe_allow_html=True)
    rampa_vdf = st.slider("Rampa VDF [rpm/s]", min_value=50, max_value=1000, value=300, step=10)

with c_b:
    st.markdown("**Cálculo**")
    # Aceleración por par: n_dot_torque [rpm/s] = (60/(2π)) * (T/J_eq)
    if J_eq and not np.isnan(J_eq) and T_disp and not np.isnan(T_disp):
        n_dot_torque = (60.0 / (2.0 * pi)) * (T_disp / J_eq)  # rpm/s
        t_par = (Delta_n / n_dot_torque) if Delta_n and not np.isnan(Delta_n) else np.nan
    else:
        n_dot_torque = np.nan
        t_par = np.nan

    t_rampa = (Delta_n / rampa_vdf) if Delta_n and not np.isnan(Delta_n) else np.nan

    st.markdown(
        f"- **Aceleración por par**: $\\dot n_{{torque}}=\\tfrac{{60}}{{2\\pi}}\\,\\tfrac{{T_{{disp}}}}{{J_{{eq}}}}$ → "
        f"{colored_value(fmt_num(n_dot_torque, 'rpm/s', 1), 'calc')}",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"- **Tiempo por par**: $t_{{par}} = \\Delta n / \\dot n_{{torque}}$ → "
        f"{colored_value(fmt_num(t_par, 's', 1), 'calc')}",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"- **Tiempo por rampa**: $t_{{rampa}} = \\Delta n / \\mathrm{{rampa}}_{{VDF}}$ → "
        f"{colored_value(fmt_num(t_rampa, 's', 1), 'calc')}",
        unsafe_allow_html=True,
    )

# Recuadro verde con el tiempo limitante
def pick_limiting_time(tp, tr):
    if (tp is None or np.isnan(tp)) and (tr is None or np.isnan(tr)):
        return None, None
    if tr is None or np.isnan(tr):
        return "Par", tp
    if tp is None or np.isnan(tp):
        return "Rampa VDF", tr
    # El limitante es el más alto (más lento)
    return ("Par", tp) if tp >= tr else ("Rampa VDF", tr)

lim_name, lim_time = pick_limiting_time(t_par, t_rampa)
if lim_name is not None:
    st.success(
        f"**Tiempo limitante:** {lim_name} → {fmt_num(lim_time, 's', 1)}",
        icon="✅",
    )
else:
    st.warning("No es posible determinar el tiempo limitante con los datos actuales.", icon="⚠️")

st.info("Próximo paso: integrar hidráulica ($Q(n)$, $H(Q)$, $\\eta(Q)$) para refinar el tiempo con carga.", icon="ℹ️")
