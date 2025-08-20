# -*- coding: utf-8 -*-
"""
Memoria de Cálculo – Tiempo de reacción (VDF)
App Streamlit con:
  • Lectura de dataset Excel `bombas_dataset_with_torque_params.xlsx`
  • Resumen de parámetros por TAG (motor, transmisión, bomba, sistema)
  • Cálculo inercial (sin e hidráulica) + tiempos (t_par, t_rampa)
  • Gráfico Plotly con 3 ejes (Q, n_bomba, P) — FIX: yaxis3.position ∈ [0,1]
  • Descargas CSV en memoria (sin rutas temporales) — evita MediaFileHandler
  • FIX warnings: raw strings en LaTeX/Markdown con \( \)

Nota: el código tolera columnas faltantes usando valores por defecto y
parsers robustos para numéricos en string.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Fallback elegante si plotly no está disponible (no debería pasar, pero por si acaso)
try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:  # pragma: no cover
    PLOTLY_OK = False

# ──────────────────────────────────────────────────────────────────────────────
# Configuración de página
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Memoria de Cálculo – Tiempo de reacción (VDF)",
    page_icon="🧮",
    layout="wide",
)

st.title("Memoria de Cálculo – Tiempo de reacción (VDF)")
st.caption(
    r"Aplicación para estimar tiempos de reacción de bombas con VDF, usando "
    r"parámetros de motor, transmisión y curva de bomba."
)

# ──────────────────────────────────────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────────────────────────────────────
_num_re = re.compile(r"[^\d\.\-eE+]")

def get_num(x, default: float = 0.0) -> float:
    """Parsea un número desde múltiples formatos (espacios, comas, unidades).
    - Si x es NaN → default
    - Si ya es número → float(x)
    - Si es string → limpia espacios, no-break-space, cambia coma por punto y
      elimina cualquier carácter no numérico (excepto ., -, e/E, +).
    """
    try:
        if x is None:
            return float(default)
        if isinstance(x, (int, float, np.number)):
            if pd.isna(x):
                return float(default)
            return float(x)
        s = str(x).strip().replace(" ", "").replace("\u00a0", "")
        s = s.replace(",", ".")
        s = _num_re.sub("", s)
        if s in ("", ".", "+", "-", "+.", "-."):
            return float(default)
        return float(s)
    except Exception:
        return float(default)


def csv_bytes_from_df(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def badge(value: str | float, unidad: str, etiqueta: str):
    """Renderiza una "badge" simple con etiqueta, valor y unidad."""
    if isinstance(value, (int, float, np.floating)):
        value_str = f"{value:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
    else:
        value_str = str(value)
    st.markdown(
        f"<div style='display:inline-block;padding:6px 10px;margin:4px;"
        f"background:#f6f6f6;border:1px solid #ddd;border-radius:10px;"
        f"font-family:ui-monospace,monospace;font-size:0.9rem;'>"
        f"<b>{etiqueta}</b>: {value_str} <span style='opacity:.7'>{unidad}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


@dataclass
class Params:
    TAG: str = "TAG"
    J_m: float = 0.0             # kg·m²
    J_driver: float = 0.0        # kg·m²
    J_driven: float = 0.0        # kg·m²
    J_imp: float = 0.0           # kg·m²
    r: float = 1.0               # relación = n_motor / n_bomba
    H0: float = 0.0              # m
    K: float = 0.0               # coef. curva m + K*(Q/3600)^2
    T_disp: float = 0.0          # N·m (torque disponible en motor)
    eta: float = 0.7             # eficiencia hidráulica

    # Nominales para escalado (opcionales)
    n_bomba_nom: float = 1500.0  # rpm
    Q_nom: float = 0.0           # m3/h a nominal (si existe)


# ──────────────────────────────────────────────────────────────────────────────
# Lectura de dataset
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_default_excel() -> Optional[pd.DataFrame]:
    try:
        return pd.read_excel("bombas_dataset_with_torque_params.xlsx")
    except Exception:
        return None


uploaded = st.file_uploader(
    "📄 Subí el dataset Excel (hoja con parámetros)", type=["xlsx", "xls"],
    help="Si no subís nada, intentaré leer 'bombas_dataset_with_torque_params.xlsx' del repo."
)

if uploaded is not None:
    try:
        df_raw = pd.read_excel(uploaded)
        st.success("Dataset cargado desde archivo subido.")
    except Exception as e:
        st.error(f"No se pudo leer el Excel subido: {e}")
        df_raw = None
else:
    df_raw = load_default_excel()
    if df_raw is not None:
        st.info("Usando dataset por defecto del repositorio.")

if df_raw is None or df_raw.empty:
    st.warning("No se encontró dataset. Se generará uno de ejemplo para probar la app.")
    df_raw = pd.DataFrame(
        {
            "TAG": ["B-101", "B-102"],
            "J_m_kgm2": [0.35, 0.50],
            "J_driver_kgm2": [0.05, 0.08],
            "J_driven_kgm2": [0.18, 0.22],
            "J_imp_kgm2": [0.07, 0.09],
            "r": [1.00, 1.00],
            "H0_m": [20.0, 25.0],
            "K_m": [-0.0004, -0.00035],  # curva típica decreciente vs Q
            "T_disp_Nm": [250.0, 300.0],
            "eta": [0.72, 0.70],
            "n_bomba_nom_rpm": [1500, 1500],
            "Q_nom_m3h": [400.0, 450.0],
        }
    )

# Normalizamos nombres (para tolerar variantes)
cols = {c.lower(): c for c in df_raw.columns}


def first_existing(*names: str) -> Optional[str]:
    for n in names:
        key = n.lower()
        if key in cols:
            return cols[key]
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Parámetros globales (sidebar)
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Ajustes de simulación")

    rampa_vdf = st.slider("Rampa VDF [rpm/s]", min_value=10, max_value=1000, value=200, step=10)
    n_ini = st.number_input("n_motor inicial [rpm]", min_value=0, max_value=6000, value=0, step=10)
    n_obj = st.number_input("n_motor objetivo [rpm]", min_value=100, max_value=6000, value=1500, step=50)

    rho = st.number_input("Densidad ρ [kg/m³]", min_value=500.0, max_value=2000.0, value=1000.0, step=10.0)
    g = 9.81
    dt = st.number_input("∆t [s]", min_value=0.005, max_value=0.5, value=0.02, step=0.005, format="%.3f")
    t_max = st.number_input("t máx [s]", min_value=1.0, max_value=120.0, value=20.0, step=1.0)

# ──────────────────────────────────────────────────────────────────────────────
# Construcción de tabla limpia + selección de TAG
# ──────────────────────────────────────────────────────────────────────────────

def row_to_params(row: pd.Series) -> Params:
    return Params(
        TAG=str(row.get(first_existing("TAG") or "TAG", "TAG")),
        J_m=get_num(row.get(first_existing("J_m_kgm2", "J_m", "Jmotor"), 0.0)),
        J_driver=get_num(row.get(first_existing("J_driver_kgm2", "J_driver"), 0.0)),
        J_driven=get_num(row.get(first_existing("J_driven_kgm2", "J_driven"), 0.0)),
        J_imp=get_num(row.get(first_existing("J_imp_kgm2", "J_imp"), 0.0)),
        r=max(get_num(row.get(first_existing("r", "ratio", "relacion"), 1.0)), 1e-6),
        H0=get_num(row.get(first_existing("H0_m", "H0"), 0.0)),
        K=get_num(row.get(first_existing("K_m", "K"), 0.0)),
        T_disp=get_num(row.get(first_existing("T_disp_Nm", "Tdisp", "Torque"), 0.0)),
        eta=np.clip(get_num(row.get(first_existing("eta"), 0.7), 0.7), 0.05, 0.99),
        n_bomba_nom=get_num(row.get(first_existing("n_bomba_nom_rpm", "n_nom_bomba", "n_bomba_nom"), 1500.0)),
        Q_nom=get_num(row.get(first_existing("Q_nom_m3h", "Q_nom"), 0.0)),
    )


params_list = [row_to_params(r) for _, r in df_raw.iterrows()]

if not params_list:
    st.stop()

TAGs = [p.TAG for p in params_list]
col_sel1, col_sel2 = st.columns([1, 2])
with col_sel1:
    sel_tag = st.selectbox("TAG", TAGs, index=0)
sel_params = next(p for p in params_list if p.TAG == sel_tag)

# ──────────────────────────────────────────────────────────────────────────────
# Cálculos base
# ──────────────────────────────────────────────────────────────────────────────
# Inercia equivalente (lado bomba reflejado al motor)
J_eq = sel_params.J_m + sel_params.J_driver + (sel_params.J_driven + sel_params.J_imp) / (sel_params.r ** 2)

st.subheader("Parámetros del sistema")
with st.expander("Ver resumen del TAG seleccionado", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        badge(sel_params.TAG, "", "TAG")
        badge(J_eq, "kg·m²", "J_eq")
    with c2:
        badge(sel_params.r, r"", "r = n_m/n_p")
        badge(sel_params.T_disp, "N·m", "T_disp")
    with c3:
        badge(sel_params.H0, "m", "H0")
        badge(sel_params.K, r"m·(h/m³)²", "K curva")
    with c4:
        badge(sel_params.eta, "-", "η")
        badge(sel_params.n_bomba_nom, "rpm", "n_bomba_nom")

st.caption(
    r"Las inercias del lado bomba giran a $\omega_p = \omega_m/r$. Igualando energías "
    r"cinéticas a una $\omega_m$ común se obtiene la división por $r^2$."
)

# Tiempos característicos (analíticos)
Delta_n = max(n_obj - n_ini, 0.0)
ndot_torque = (60.0 / (2.0 * np.pi)) * (sel_params.T_disp / max(J_eq, 1e-9))  # rpm/s si T_disp limita

t_par = Delta_n / max(ndot_torque, 1e-9) if sel_params.T_disp > 0 else np.nan

t_rampa = Delta_n / max(rampa_vdf, 1e-9)

with st.expander("Fórmulas empleadas", expanded=False):
    st.latex(r"J_{\mathrm{eq}} = J_m + J_{\mathrm{driver}} + \dfrac{J_{\mathrm{driven}}+J_{\mathrm{imp}}}{r^2}")
    st.markdown(r"**Ecuaciones base:** $\dot n_{\mathrm{torque}}=\dfrac{60}{2\pi}\dfrac{T_{disp}}{J_{eq}}$, "
                r"$t_{par}=\dfrac{\Delta n}{\dot n_{\mathrm{torque}}}$, "
                r"$t_{rampa}=\dfrac{\Delta n}{\mathrm{rampa}_{VDF}}$. ")
    st.markdown(r"**Con hidráulica:** $J_{eq}\,\dot\omega_m=T_{disp}-T_{pump}/r$, "
                r"$T_{pump}=\dfrac{\rho g\,Q\,H(Q)}{\eta\,\omega_p}$, $\omega_p=\omega_m/r$, $Q\propto n_p$.")

st.subheader("Tiempos característicos")
cA, cB, cC = st.columns(3)
with cA: badge(ndot_torque, "rpm/s", "Ṅ_torque")
with cB: badge(t_par, "s", "t_par")
with cC: badge(t_rampa, "s", "t_rampa")

# ──────────────────────────────────────────────────────────────────────────────
# Simulación simple (rampa impuesta en velocidad; hidráulica "estática")
# ──────────────────────────────────────────────────────────────────────────────
# Para claridad, imponemos n_m(t) por rampa (limitada por objetivo) y derivamos Q,P,H con curvas.
T = np.arange(0.0, t_max + 1e-9, dt)
# rampa lineal del motor
n_m = np.minimum(n_ini + rampa_vdf * T, n_obj)
# velocidad bomba
n_p = n_m / max(sel_params.r, 1e-9)
# caudal ~ proporcional a n_p (si no hay Q_nom, escalamos a 1 m3/h por 100 rpm como dummy)
if sel_params.Q_nom > 0 and sel_params.n_bomba_nom > 0:
    Q = (n_p / sel_params.n_bomba_nom) * sel_params.Q_nom
else:
    Q = n_p * 0.01  # 0.01 m3/h por rpm como suposición básica

# Curva de bomba H(Q) = H0 + K*(Q/3600)^2
H = sel_params.H0 + sel_params.K * (Q / 3600.0) ** 2
# Potencia hidráulica P = rho*g*Q*H  [W]; convertimos a kW
P_kW = (rho * g * (Q / 3600.0) * H) / 1000.0

# ──────────────────────────────────────────────────────────────────────────────
# Gráfico Plotly (3 ejes) – FIX: yaxis3.position dentro de [0,1]
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("Respuesta temporal (rampa VDF)")

if PLOTLY_OK:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=T, y=Q, name="Q [m³/h]", yaxis="y"))
    fig.add_trace(go.Scatter(x=T, y=n_p, name="n bomba [rpm]", yaxis="y2"))
    fig.add_trace(go.Scatter(x=T, y=P_kW, name="P hidráulica [kW]", yaxis="y3"))

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        xaxis=dict(title="Tiempo [s]"),
        yaxis=dict(title="Q [m³/h]", anchor="x"),  # izquierda
        yaxis2=dict(title="n bomba [rpm]", overlaying="y", side="right", anchor="x"),
        # Tercer eje a la derecha con anchor='free' y position ∈ [0,1]
        yaxis3=dict(title="P [kW]", overlaying="y", side="right", anchor="free", position=0.98),
    )

    st.plotly_chart(fig, use_container_width=True)
else:  # Fallback a Matplotlib (mínimo)
    import matplotlib.pyplot as plt  # type: ignore

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax2 = ax1.twinx()
    ax1.plot(T, Q, label="Q [m³/h]")
    ax2.plot(T, n_p, label="n bomba [rpm]")
    ax1.set_xlabel("Tiempo [s]")
    ax1.set_ylabel("Q [m³/h]")
    ax2.set_ylabel("n bomba [rpm]")
    fig.tight_layout()
    st.pyplot(fig)

# ──────────────────────────────────────────────────────────────────────────────
# Reportes CSV (en memoria) – evita MediaFileHandler
# ──────────────────────────────────────────────────────────────────────────────

def resumen_por_params(p: Params) -> dict:
    r_local = max(p.r, 1e-9)
    J_eq_local = p.J_m + p.J_driver + (p.J_driven + p.J_imp) / (r_local ** 2)
    ndot = (60.0 / (2.0 * np.pi)) * (p.T_disp / max(J_eq_local, 1e-9)) if p.T_disp > 0 else np.nan
    tpar = (Delta_n / ndot) if (p.T_disp > 0 and ndot > 0) else np.nan
    tramp = Delta_n / max(rampa_vdf, 1e-9)
    return {
        "TAG": p.TAG,
        "J_eq_kgm2": J_eq_local,
        "r_nmotor_nbomba": r_local,
        "T_disp_Nm": p.T_disp,
        "ndot_torque_rpmps": ndot,
        "t_par_s": tpar,
        "t_rampa_s": tramp,
        "H0_m": p.H0,
        "K_m": p.K,
        "eta": p.eta,
        "n_bomba_nom_rpm": p.n_bomba_nom,
        "Q_nom_m3h": p.Q_nom,
    }

# Reporte del TAG seleccionado
res_sel = pd.DataFrame([resumen_por_params(sel_params)])
col_dl1, col_dl2 = st.columns([1, 2])
with col_dl1:
    st.download_button(
        "⬇️ Descargar reporte (TAG seleccionado)",
        data=csv_bytes_from_df(res_sel),
        file_name=f"reporte_{sel_params.TAG}_rampa_{int(rampa_vdf)}rpmps.csv",
        mime="text/csv",
    )

# Reporte de todos los TAG
res_all = pd.DataFrame([resumen_por_params(p) for p in params_list])
st.download_button(
    "⬇️ Descargar reporte (todos los TAG, con rampa seleccionada)",
    data=csv_bytes_from_df(res_all),
    file_name=f"reporte_todos_los_TAG_rampa_{int(rampa_vdf)}rpmps.csv",
    mime="text/csv",
)

st.markdown("---")

# Mostrar tabla base (opcional)
with st.expander("Ver tabla original del dataset", expanded=False):
    st.dataframe(df_raw, use_container_width=True)
