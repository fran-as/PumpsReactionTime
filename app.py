# app.py
import math
import re
from io import StringIO
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

# ===============================
# Utilidades
# ===============================

def to_float(x) -> float:
    """Convierte strings (coma/punto) a float. NaN -> 0.0"""
    if pd.isna(x):
        return 0.0
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace(" ", "").replace("\u00a0", "")
    s = s.replace(",", ".")
    s = re.sub(r"[^\d\.\-eE+]", "", s)
    try:
        return float(s)
    except Exception:
        return 0.0

def fmt(x, unit=""):
    """2 decimales con coma decimal."""
    try:
        v = float(x)
    except Exception:
        return str(x)
    s = f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{s} {unit}".strip()

def find_first_column(df: pd.DataFrame, *patterns: str) -> Optional[str]:
    """Retorna el primer nombre de columna que cumpla TODOS los patrones regex (case-insensitive)."""
    pats = [re.compile(p, re.I) for p in patterns]
    for c in df.columns:
        if all(p.search(str(c)) for p in pats):
            return c
    return None

def sum_if_exists(row: pd.Series, cols: List[str]) -> float:
    total = 0.0
    for c in cols:
        if c in row.index:
            total += to_float(row[c])
    return total

# ===============================
# Config
# ===============================
st.set_page_config(page_title="Memoria de Cálculo – Tiempo de reacción de bombas (VDF)", layout="wide")

st.title("Memoria de Cálculo – Tiempo de reacción de bombas (VDF)")

DATA_FILE = "bombas_dataset_with_torque_params.xlsx"

@st.cache_data(show_spinner=True)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    return df

try:
    df = load_data(DATA_FILE)
except Exception as e:
    st.error(f"No pude abrir **{DATA_FILE}**. Verifica que esté en la raíz. Detalle: {e}")
    st.stop()

# Por acuerdo: la PRIMERA columna es el TAG y es única
COL_TAG = df.columns[0]
TAGS = df.iloc[:, 0].astype(str).tolist()

# ===============================
# Mapeo flexible de columnas
# ===============================
def column_map(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    m = {}
    # Inercias
    m["J_m"]   = find_first_column(df, r"motor", r"J", r"kgm") or "J_m_kgm2"
    m["J_imp"] = find_first_column(df, r"impell|impuls", r"J", r"kgm") or "J_imp_kgm2"

    m["J_drv_pul"] = find_first_column(df, r"driver", r"pul") or "J_driver_pulley_kgm2"
    m["J_drv_bus"] = find_first_column(df, r"driver", r"(bush|mangui)") or "J_driver_bushing_kgm2"

    m["J_drn_pul"] = find_first_column(df, r"driven", r"pul") or "J_driven_pulley_kgm2"
    m["J_drn_bus"] = find_first_column(df, r"driven", r"(bush|mangui)") or "J_driven_bushing_kgm2"

    # Relación
    m["ratio"] = (find_first_column(df, r"relaci[óo]n", r"trans")
                  or find_first_column(df, r"ratio")
                  or "ratio")

    # Velocidades motor
    m["n_min"] = find_first_column(df, r"motor", r"min", r"rpm") or "motor_n_min_rpm"
    m["n_max"] = find_first_column(df, r"motor", r"max", r"rpm") or "motor_n_max_rpm"
    m["n_nom"] = find_first_column(df, r"motor", r"(nom|max)", r"rpm") or m["n_max"]

    # Par o potencia
    m["T_nom"] = find_first_column(df, r"T.*nom.*Nm")
    m["P_kw"]  = find_first_column(df, r"motor", r"(power|poten)", r"kW")

    return m

MAP = column_map(df)

def read_row_values(row: pd.Series) -> Dict[str, float]:
    J_m    = to_float(row.get(MAP["J_m"], 0.0))
    J_imp  = to_float(row.get(MAP["J_imp"], 0.0))
    J_drv  = to_float(row.get(MAP["J_drv_pul"], 0.0)) + to_float(row.get(MAP["J_drv_bus"], 0.0))
    J_drn  = to_float(row.get(MAP["J_drn_pul"], 0.0)) + to_float(row.get(MAP["J_drn_bus"], 0.0))
    r_tr   = max(to_float(row.get(MAP["ratio"], 1.0)), 1e-9)
    n_min  = to_float(row.get(MAP["n_min"], 0.0))
    n_max  = to_float(row.get(MAP["n_max"], 0.0))
    n_nom  = to_float(row.get(MAP["n_nom"], n_max if n_max > 0 else 1500.0))

    # Par disponible
    if MAP["T_nom"] and MAP["T_nom"] in row.index and to_float(row[MAP["T_nom"]]) > 0:
        T_nom = to_float(row[MAP["T_nom"]])
    else:
        P_kw = to_float(row.get(MAP["P_kw"], 0.0))
        T_nom = 9550.0 * P_kw / max(n_nom, 1e-9)  # T [Nm] = 9550 P[kW]/n[rpm]

    return dict(J_m=J_m, J_imp=J_imp, J_drv=J_drv, J_drn=J_drn, r_tr=r_tr,
                n_min=n_min, n_max=n_max, n_nom=n_nom, T_nom=T_nom)

def compute_J_eq(J_m, J_drv, J_drn, J_imp, r_tr):
    return J_m + J_drv + (J_drn + J_imp) / (r_tr ** 2)

def compute_times(J_eq, T_disp, n_ini, n_fin, rampa_vdf):
    dn = max(n_fin - n_ini, 0.0)
    n_dot_torque = (60.0 / (2.0 * math.pi)) * (T_disp / max(J_eq, 1e-9))  # rpm/s
    t_par = dn / max(n_dot_torque, 1e-9)
    t_rampa = dn / max(rampa_vdf, 1e-9)
    t_final = max(t_par, t_rampa)
    return dict(delta_n=dn, n_dot=n_dot_torque, t_par=t_par, t_rampa=t_rampa, t_final=t_final)

# ===============================
# UI – Selección
# ===============================
st.sidebar.header("Selección")
tag = st.sidebar.selectbox("TAG", TAGS, index=0)
row = df.loc[df[COL_TAG].astype(str) == tag]
if row.empty:
    st.error("No encontré el TAG seleccionado en el dataset.")
    st.stop()
row = row.iloc[0]

vals = read_row_values(row)
J_m, J_imp, J_drv, J_drn, r_tr = vals["J_m"], vals["J_imp"], vals["J_drv"], vals["J_drn"], vals["r_tr"]
n_min, n_max, n_nom, T_nom = vals["n_min"], vals["n_max"], vals["n_nom"], vals["T_nom"]

# ===============================
# 1) Datos de entrada
# ===============================
st.subheader("1) Datos de entrada (por TAG)")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("**TAG**")
    st.markdown(f"{tag}")
with c2:
    st.markdown("**Relación de transmisión**  \n\(r = n_{motor}/n_{bomba}\)")
    st.markdown(f"{fmt(r_tr)}")
with c3:
    st.markdown("**Velocidad Motor min–max [rpm]**")
    st.markdown(f"{fmt(n_min,'rpm')} – {fmt(n_max,'rpm')}")
with c4:
    st.markdown(r"**Par disponible \(T_{\mathrm{nom}}\) [Nm]**")
    st.markdown(f"{fmt(T_nom,'Nm')}")

st.markdown("---")

# ===============================
# 2) Inercia equivalente al eje del motor
# ===============================
st.subheader("2) Inercia equivalente al eje del motor")

st.latex(r"""
J_{\mathrm{eq}} \;=\; J_m \;+\; J_{\mathrm{driver}} \;+\; \frac{J_{\mathrm{driven}} + J_{\mathrm{imp}}}{r^{2}}
""")

exp = st.expander("¿Por qué dividir por \(r^2\)?")
with exp:
    st.latex(r"""
\text{Las inercias del lado bomba giran a } \omega_p = \omega_m / r.
""")
    st.latex(r"""
\frac{1}{2}J_{\mathrm{eq}}\,\omega_m^2
= \frac{1}{2}J_m\,\omega_m^2
+ \frac{1}{2}J_{\mathrm{driver}}\,\omega_m^2
+ \frac{1}{2}J_{\mathrm{driven}}\,\omega_p^2
+ \frac{1}{2}J_{\mathrm{imp}}\,\omega_p^2
\;\Rightarrow\;
J_{\mathrm{eq}} = J_m + J_{\mathrm{driver}} + \frac{J_{\mathrm{driven}}+J_{\mathrm{imp}}}{r^2}.
""")

cI1, cI2, cI3, cI4, cI5 = st.columns(5)
with cI1:
    st.markdown("**J_m (motor)**")
    st.success(fmt(J_m, "kg·m²"))
with cI2:
    st.markdown("**J_driver (total)**  \nL = polea + manguito")
    st.success(fmt(J_drv, "kg·m²"))
with cI3:
    st.markdown("**J_driven (total)**  \nL = polea + manguito")
    st.success(fmt(J_drn, "kg·m²"))
with cI4:
    st.markdown("**J_imp (impulsor)**")
    st.success(fmt(J_imp, "kg·m²"))

J_eq = compute_J_eq(J_m, J_drv, J_drn, J_imp, r_tr)
with cI5:
    st.markdown("**Total \(J_{eq}\)**")
    st.info(fmt(J_eq, "kg·m²"))

# Sustitución numérica en LaTeX
st.latex(rf"""
\begin{{aligned}}
J_{{\mathrm{{eq}}}} &= J_m + J_{{\mathrm{{driver}}}} + \frac{{J_{{\mathrm{{driven}}}} + J_{{\mathrm{{imp}}}}}}{{r^2}}\\
&= {J_m:.2f} + {J_drv:.2f} + \frac{{{J_drn:.2f}+{J_imp:.2f}}}{{({r_tr:.2f})^2}}\\
&= \mathbf{{{J_eq:.2f}}}\ \mathrm{{kg\cdot m^2}}
\end{{aligned}}
""")

st.markdown("---")

# ===============================
# 3) Tiempo de reacción SIN hidráulica
# ===============================
st.subheader("3) Tiempo de reacción sin hidráulica")

st.latex(r"""
\dot n_{\mathrm{torque}} = \frac{60}{2\pi}\,\frac{T_{\mathrm{nom}}}{J_{\mathrm{eq}}},\quad
t_{\mathrm{par}}=\frac{\Delta n}{\dot n_{\mathrm{torque}}},\quad
t_{\mathrm{rampa}}=\frac{\Delta n}{\mathrm{rampa}_{\mathrm{VDF}}},\quad
t_{\mathrm{final,\,sin}}=\max\{t_{\mathrm{par}},\,t_{\mathrm{rampa}}\}.
""")

c3a, c3b, c3c = st.columns(3)
with c3a:
    n_ini = st.number_input("Velocidad Motor inicial [rpm]", value=float(n_min))
with c3b:
    n_fin = st.number_input("Velocidad Motor final [rpm]", value=float(n_max if n_max > 0 else n_min + 300))
with c3c:
    T_disp = st.number_input(r"Par disponible \(T_{\mathrm{nom}}\) [Nm]", value=float(T_nom))

rampa_vdf = st.sidebar.number_input("Rampa VDF [rpm/s] (motor)", min_value=1.0, value=300.0, step=1.0)

times = compute_times(J_eq, T_disp, n_ini, n_fin, rampa_vdf)
delta_n, n_dot_torque, t_par, t_rampa, t_final_sin = times["delta_n"], times["n_dot"], times["t_par"], times["t_rampa"], times["t_final"]

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Δn [rpm]", fmt(delta_n, "rpm"))
with m2:
    st.metric(r"\(\dot n_{\mathrm{torque}}\) [rpm/s]", fmt(n_dot_torque, "rpm/s"))
with m3:
    st.metric("t_final (sin hidráulica) [s]", fmt(t_final_sin, "s"))

# “Franja verde” con LaTeX (usamos st.success + st.latex)
st.success("Resumen de cálculo (sin hidráulica)")
st.latex(rf"""
\Delta n = {delta_n:.2f}\ \mathrm{{rpm}}
\ \Big|\ 
\dot n_{{\mathrm{{torque}}}} = {n_dot_torque:.2f}\ \mathrm{{rpm/s}}
\ \Rightarrow\ 
t_{{\mathrm{{par}}}}={t_par:.2f}\ \mathrm{{s}},\quad
t_{{\mathrm{{rampa}}}}={t_rampa:.2f}\ \mathrm{{s}},\quad
\mathbf{{t_{{\mathrm{{final,\,sin}}}}={t_final_sin:.2f}\ \mathrm{{s}}}}
""")

st.caption("Nota: aquí aún no se incluye el par hidráulico de la bomba ni la curva del sistema; sólo inercia mecánica y rampa del VDF.")

st.markdown("---")

# ===============================
# 4) Hidráulica (formulación)
# ===============================
st.subheader("4) Siguiente paso: incluir hidráulica")

st.latex(r"""
H(Q)=H_0+K\,Q^2,\qquad
P_h=\rho g Q H,\qquad
T_{\mathrm{load}}=\frac{P_h}{\omega}=\frac{\rho g\,Q\,(H_0+KQ^2)}{2\pi n/60}.
""")
st.markdown(
    "- Ajustar \(H_0\) y \(K\) con los 5–6 puntos de operación por TAG.\n"
    "- Integrar la ecuación de movimiento \(J_{eq}\,\dot\omega=T_{\mathrm{disp}}-T_{\mathrm{load}}(\omega)\) para obtener el **tiempo con hidráulica**.\n"
    "- Usar \(\\rho = \\text{SG} \\cdot 1000\\ \\mathrm{kg/m^3}\) por TAG para el par resistente."
)

st.markdown("---")

# ===============================
# 5) Exportar tabla por TAG (dataset completo)
# ===============================
st.subheader("5) Exportar resumen por TAG (rampa actual)")

st.markdown(
    "Genera una tabla con **inercias por componente**, \(J_{eq}\), **Δn**, "
    r"\(\dot n_{\mathrm{torque}}\), \(t_{\mathrm{par}}\), \(t_{\mathrm{rampa}}\) y \(t_{\mathrm{final}}\) "
    "para **cada TAG**, usando la **rampa** indicada en la barra lateral."
)

def build_summary_table(df: pd.DataFrame, rampa_vdf: float) -> pd.DataFrame:
    rows = []
    for _, rw in df.iterrows():
        vals_i = read_row_values(rw)
        J_m_i, J_imp_i, J_drv_i, J_drn_i, r_tr_i = vals_i["J_m"], vals_i["J_imp"], vals_i["J_drv"], vals_i["J_drn"], vals_i["r_tr"]
        n_min_i, n_max_i, T_nom_i = vals_i["n_min"], vals_i["n_max"], vals_i["T_nom"]
        J_eq_i = compute_J_eq(J_m_i, J_drv_i, J_drn_i, J_imp_i, r_tr_i)
        times_i = compute_times(J_eq_i, T_nom_i, n_min_i, n_max_i, rampa_vdf)
        rows.append({
            "TAG": str(rw[COL_TAG]),
            "J_m (kg·m²)": J_m_i,
            "J_driver (kg·m²)": J_drv_i,
            "J_driven (kg·m²)": J_drn_i,
            "J_imp (kg·m²)": J_imp_i,
            "r (n_motor/n_bomba)": r_tr_i,
            "J_eq (kg·m²)": J_eq_i,
            "n_motor min [rpm]": n_min_i,
            "n_motor max [rpm]": n_max_i,
            "Δn [rpm]": times_i["delta_n"],
            "n_dot_torque [rpm/s]": times_i["n_dot"],
            "t_par [s]": times_i["t_par"],
            "t_rampa [s]": times_i["t_rampa"],
            "t_final_sin [s]": times_i["t_final"],
        })
    out = pd.DataFrame(rows)
    # redondeo a 2 decimales
    num_cols = [c for c in out.columns if c != "TAG"]
    out[num_cols] = out[num_cols].astype(float).round(2)
    return out

if st.button("Generar tabla"):
    tbl = build_summary_table(df, rampa_vdf)
    st.dataframe(tbl, use_container_width=True)
    csv = tbl.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Descargar CSV", data=csv, file_name="resumen_por_TAG.csv", mime="text/csv")
