# app.py
import math
import re
from typing import Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st

# -------- utilidades --------
def to_float(x) -> float:
    """Convierte strings con coma/punto a float; NaN -> 0.0"""
    if pd.isna(x):
        return 0.0
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip()
    s = s.replace(" ", "").replace("\u00a0", "")
    # decimal comma -> dot
    s = s.replace(",", ".")
    # quitar sufijos como "kg·m2", "kgm^2", etc., y unidades
    s = re.sub(r"[^\d\.\-eE+]", "", s)
    try:
        return float(s)
    except Exception:
        return 0.0

def fmt(x, unit=""):
    """Muestra con 2 decimales y coma decimal."""
    try:
        v = float(x)
    except Exception:
        return str(x)
    s = f"{v:,.2f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{s} {unit}".strip()

def find_first_column(df: pd.DataFrame, *patterns: str) -> Optional[str]:
    """
    Devuelve el primer nombre de columna cuyo nombre cumple TODOS los patterns (regex, case-insensitive).
    """
    pats = [re.compile(p, re.I) for p in patterns]
    for c in df.columns:
        if all(p.search(str(c)) for p in pats):
            return c
    return None

def sum_columns(df_row: pd.Series, candidates: list[str]) -> float:
    """Suma columnas si existen (en kg·m²)."""
    total = 0.0
    for c in candidates:
        if c in df_row.index:
            total += to_float(df_row[c])
    return total

st.set_page_config(page_title="Memoria de Cálculo – Tiempo de reacción de bombas (VDF)", layout="wide")

st.title("Memoria de Cálculo – Tiempo de reacción de bombas (VDF)")

# -------- carga del dataset --------
DATA_FILE = "bombas_dataset_with_torque_params.xlsx"

@st.cache_data(show_spinner=True)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    # normaliza encabezados
    df.columns = [str(c).strip() for c in df.columns]
    return df

try:
    df = load_data(DATA_FILE)
except Exception as e:
    st.error(f"No pude abrir **{DATA_FILE}**. Verifica que esté en la raíz. Detalle: {e}")
    st.stop()

# 1) Selección de TAG: la primera columna es el TAG y es único
col_tag = df.columns[0]
tags = df.iloc[:, 0].astype(str).tolist()
st.sidebar.header("Selección")
tag = st.sidebar.selectbox("TAG", tags, index=0)

row = df.loc[df[col_tag].astype(str) == tag]
if row.empty:
    st.error("No encontré el TAG seleccionado en el dataset.")
    st.stop()
row = row.iloc[0]  # Series

# -------- mapeo de columnas por patrones (robusto a nombres) --------
col_Jm   = find_first_column(df, r"motor", r"J", r"kgm") or "J_m_kgm2"
col_Jimp = find_first_column(df, r"impell", r"J", r"kgm") or "J_imp_kgm2"

# partes driver (lado motor)
col_Jdrv_pul = find_first_column(df, r"driver", r"pul") or "J_driver_pulley_kgm2"
col_Jdrv_bus = find_first_column(df, r"driver", r"bush|mangui") or "J_driver_bushing_kgm2"

# partes driven (lado bomba)
col_Jdrn_pul = find_first_column(df, r"driven", r"pul") or "J_driven_pulley_kgm2"
col_Jdrn_bus = find_first_column(df, r"driven", r"bush|mangui") or "J_driven_bushing_kgm2"

# relación de transmisión r = n_motor / n_bomba
col_ratio = (find_first_column(df, r"relaci[óo]n", r"trans")
             or find_first_column(df, r"ratio")
             or "ratio")

# velocidades motor mín/máx (si existen)
col_n_min = find_first_column(df, r"motor", r"min", r"rpm") or "motor_n_min_rpm"
col_n_max = find_first_column(df, r"motor", r"max", r"rpm") or "motor_n_max_rpm"

# potencia para par nominal (o par directo si existe)
col_T_nom = find_first_column(df, r"T.*nom.*Nm")  # si ya viene
col_P_kw  = find_first_column(df, r"motor", r"power|poten", r"kW")  # si hay que calcular
col_n_nom = find_first_column(df, r"motor", r"(max|nom)", r"rpm")  # velocidad para T_nom

# -------- lee valores numéricos --------
J_m    = to_float(row.get(col_Jm, 0.0))
J_imp  = to_float(row.get(col_Jimp, 0.0))
J_drv  = to_float(row.get(col_Jdrv_pul, 0.0)) + to_float(row.get(col_Jdrv_bus, 0.0))
J_drn  = to_float(row.get(col_Jdrn_pul, 0.0)) + to_float(row.get(col_Jdrn_bus, 0.0))
r_tr   = max(to_float(row.get(col_ratio, 1.0)), 1e-9)  # evita división por cero
n_min  = to_float(row.get(col_n_min, 0.0))
n_max  = to_float(row.get(col_n_max, 0.0))

# Par nominal disponible
if col_T_nom and col_T_nom in row.index and to_float(row[col_T_nom]) > 0:
    T_nom = to_float(row[col_T_nom])
else:
    P_kw  = to_float(row.get(col_P_kw, 0.0))
    n_nom = to_float(row.get(col_n_nom, n_max if n_max > 0 else 1500.0))
    # T [Nm] = 9550 * P[kW] / n[rpm]
    T_nom = 9550.0 * P_kw / max(n_nom, 1e-9)

# -------- Sección 1: Datos de entrada --------
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
    st.markdown("**Par disponible \(T_{nom}\) [Nm]**")
    st.markdown(f"{fmt(T_nom,'Nm')}")

st.markdown("---")

# -------- Sección 2: Inercia equivalente --------
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
\;\;\Rightarrow\;\;
J_{\mathrm{eq}} = J_m + J_{\mathrm{driver}} + \frac{J_{\mathrm{driven}}+J_{\mathrm{imp}}}{r^2}.
""")

# panel de inercia por componentes
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

J_eq = J_m + J_drv + (J_drn + J_imp) / (r_tr ** 2)
with cI5:
    st.markdown("**Total \(J_{eq}\)**")
    st.info(fmt(J_eq, "kg·m²"))

st.markdown("---")

# -------- Sección 3: Tiempo de reacción SIN hidráulica --------
st.subheader("3) Tiempo de reacción sin hidráulica")

st.latex(r"""
\dot n_{\mathrm{torque}} = \frac{60}{2\pi}\,\frac{T_{\mathrm{nom}}}{J_{\mathrm{eq}}},\qquad
t_{\mathrm{par}}=\frac{\Delta n}{\dot n_{\mathrm{torque}}},\qquad
t_{\mathrm{rampa}}=\frac{\Delta n}{\mathrm{rampa}_{\mathrm{VDF}}},\qquad
t_{\mathrm{final,\,sin}}=\max\{t_{\mathrm{par}},\,t_{\mathrm{rampa}}\}.
""")

c3a, c3b, c3c = st.columns(3)
with c3a:
    n_ini = st.number_input("Velocidad Motor inicial [rpm]", value=float(n_min))
with c3b:
    n_fin = st.number_input("Velocidad Motor final [rpm]", value=float(n_max if n_max > 0 else n_ini + 300))
with c3c:
    T_disp = st.number_input(r"Par disponible \(T_{nom}\) [Nm]", value=float(T_nom))

delta_n = max(n_fin - n_ini, 0.0)
rampa_vdf = st.sidebar.number_input("Rampa VDF [rpm/s] (motor)", min_value=1.0, value=300.0, step=1.0)

# cálculos
n_dot_torque = (60.0 / (2.0 * math.pi)) * (T_disp / max(J_eq, 1e-9))  # rpm/s
t_par = delta_n / max(n_dot_torque, 1e-9)
t_rampa = delta_n / max(rampa_vdf, 1e-9)
t_final_sin = max(t_par, t_rampa)

# muestra métrica
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Δn [rpm]", fmt(delta_n, "rpm"))
with m2:
    st.metric(r"\(\dot n_{\mathrm{torque}}\) [rpm/s]", fmt(n_dot_torque, "rpm/s"))
with m3:
    st.metric("t_final (sin hidráulica) [s]", fmt(t_final_sin, "s"))

st.markdown(
    r"""
<div style="padding:12px;background-color:#163b2f;border-radius:10px;color:#fff;">
\[
\Delta n = %s\ \mathrm{rpm}
\ \big|\ 
\dot n_{\mathrm{torque}} = %s\ \mathrm{rpm/s}
\ \Rightarrow\ 
t_{\mathrm{par}} = %s\ \mathrm{s},\quad
t_{\mathrm{rampa}} = %s\ \mathrm{s},\quad
\mathbf{t_{\mathrm{final,\,sin}} = %s\ \mathrm{s}}
\]
</div>
"""
    % (
        fmt(delta_n, "").replace(",", "."),
        fmt(n_dot_torque, "").replace(",", "."),
        fmt(t_par, "").replace(",", "."),
        fmt(t_rampa, "").replace(",", "."),
        fmt(t_final_sin, "").replace(",", "."),
    ),
    unsafe_allow_html=True,
)

st.caption(
    "Nota: en esta sección **no** se considera aún el par hidráulico de la bomba ni la curva del sistema; "
    "sólo inercia mecánica y rampa del VDF."
)

st.markdown("---")

# -------- Sección 4: (placeholder) hidráulica --------
st.subheader("4) Siguiente paso: incluir hidráulica")
st.markdown(
    r"""
Para integrar la hidráulica, se necesita una ley de par resistente \(T_{\mathrm{load}}(n)\) coherente con la curva del sistema.
Una forma práctica con tus 6 puntos es ajustar:
\[
H(Q)=H_0+K\,Q^2,\qquad
P_h=\rho g Q H,\qquad
T_{\mathrm{load}}=\frac{P_h}{\omega}=\frac{\rho g Q (H_0+KQ^2)}{2\pi n/60}.
\]
Con eso se integra la ecuación de movimiento \(\,J_{\mathrm{eq}}\,\dot\omega=T_{\mathrm{disp}}-T_{\mathrm{load}}(\omega)\,\)
y se obtiene el **tiempo de reacción con hidráulica**.
"""
)
