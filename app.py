# app.py
# ──────────────────────────────────────────────────────────────────────────────
# Memoria de Cálculo – Tiempo de reacción de bombas (VDF)
# App “todo-en-uno” con:
#  1) Lectura del dataset `bombas_dataset_with_torque_params.xlsx`
#  2) Resumen de parámetros por TAG (motor, transmisión, bomba, sistema)
#  3) Respuesta inercial + VDF (fórmulas LaTeX y métricas clave)
#  4) Respuesta con hidráulica (modelo sencillo): integra J_eq·ω̇ = T_disp − T_pump/r
#  5) Gráfico interactivo (Plotly) con 3 ejes y descarga de reportes por bytes
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

# Plotly (si fallara la import, mostramos un mensaje y seguimos con cálculo)
try:
    import plotly.graph_objects as go
except Exception as e:
    go = None

# ──────────────────────────────────────────────────────────────────────────────
# Configuración de página
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Memoria de Cálculo – Tiempo de reacción (VDF)",
    page_icon="⏱️",
    layout="wide",
)

st.title("⏱️ Tiempo de reacción de bombas con VDF")
st.write(
    "Herramienta para estimar tiempos de reacción considerando inercia, rampa VDF "
    "y carga hidráulica simplificada."
)

# ──────────────────────────────────────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────────────────────────────────────

def get_num(x: Any) -> float:
    """Convierte texto/num a float de forma robusta (admite coma decimal, espacios, etc.)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0.0
    if isinstance(x, (int, float, np.floating)):
        return float(x)
    s = str(x).strip().replace(" ", "").replace("\u00a0", "")
    s = s.replace(",", ".")
    s = re.sub(r"[^\d\.\-eE+]", "", s)
    try:
        return float(s)
    except Exception:
        return 0.0


def csv_bytes_from_df(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def show_badge(value_text: str, unit: str, label: str) -> None:
    """Badge básico en HTML (inline)."""
    st.markdown(
        f"""
        <div style="
            display:inline-block;
            padding:6px 10px;
            margin:4px 6px 4px 0;
            border-radius:12px;
            background:#f0f2f6;
            border:1px solid #e0e3e8;
            font-size:0.9rem;">
            <strong>{label}:</strong> {value_text} <span style="opacity:.7">{unit}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


@dataclass
class TagParams:
    TAG: str
    # Motor / transmisión / bomba
    J_m: float           # [kg·m²] Inercia motor
    J_driver: float      # [kg·m²] Inercia transmisión (acopl./reductor, lado motor)
    J_driven: float      # [kg·m²] Inercia eje bomba (lado bomba)
    J_imp: float         # [kg·m²] Inercia impulsor adicional (lado bomba)
    n_motor_nom: float   # [rpm] nominal motor
    n_bomba_nom: float   # [rpm] nominal bomba
    P_motor_kw: float    # [kW] potencia nominal motor (para estimar T_disp si no hay)
    T_disp: float        # [Nm] par disponible (constante simplificado)
    # Curva bomba y sistema
    Q_nom_m3h: float     # [m³/h] caudal nominal
    H0_m: float          # [m] intersección con eje H
    K: float             # [-] coef. cuadrático (si Q en m³/h, ver fórmula)
    eta_h: float         # [-] eficiencia hidráulica global
    rho_kgm3: float      # [kg/m³] densidad
    Q_min_m3h: float     # [m³/h] clamp inferior
    Q_max_m3h: float     # [m³/h] clamp superior

    @property
    def r(self) -> float:
        """Relación r = n_motor / n_bomba."""
        n_b = self.n_bomba_nom if self.n_bomba_nom > 0 else 1.0
        return max(self.n_motor_nom, 1e-6) / n_b

    @property
    def J_eq(self) -> float:
        """Inercia equivalente referida al eje del motor."""
        return self.J_m + self.J_driver + (self.J_driven + self.J_imp) / max(self.r**2, 1e-9)


# ──────────────────────────────────────────────────────────────────────────────
# Carga de datos
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_default_df() -> pd.DataFrame:
    # DF de ejemplo si el archivo no está; columnas esperadas.
    data = [
        {
            "TAG": "PU-101",
            "J_m": 0.45,
            "J_driver": 0.05,
            "J_driven": 0.30,
            "J_imp": 0.15,
            "n_motor_nom": 1500.0,
            "n_bomba_nom": 1000.0,
            "P_motor_kw": 90.0,
            "T_disp": 0.0,  # 0 => estimar desde P y n
            "Q_nom_m3h": 250.0,
            "H0_m": 30.0,
            "K": 0.002,     # si Q está en m³/h se usa (Q/3600)^2
            "eta_h": 0.72,
            "rho_kgm3": 1000.0,
            "Q_min_m3h": 20.0,
            "Q_max_m3h": 450.0,
        },
        {
            "TAG": "PU-102",
            "J_m": 0.60,
            "J_driver": 0.06,
            "J_driven": 0.28,
            "J_imp": 0.12,
            "n_motor_nom": 1800.0,
            "n_bomba_nom": 1200.0,
            "P_motor_kw": 110.0,
            "T_disp": 0.0,
            "Q_nom_m3h": 300.0,
            "H0_m": 28.0,
            "K": 0.0025,
            "eta_h": 0.70,
            "rho_kgm3": 1000.0,
            "Q_min_m3h": 30.0,
            "Q_max_m3h": 500.0,
        },
    ]
    return pd.DataFrame(data)


@st.cache_data
def load_dataset(upload: io.BytesIO | None) -> pd.DataFrame:
    # Si subieron archivo, lo usamos; si no, intentamos ruta por defecto; si no, DF ejemplo.
    if upload is not None:
        try:
            return pd.read_excel(upload)
        except Exception:
            upload.seek(0)
            try:
                return pd.read_csv(upload)
            except Exception:
                pass

    # Intento de ruta por defecto del repo
    try:
        return pd.read_excel("bombas_dataset_with_torque_params.xlsx")
    except Exception:
        return load_default_df()


def row_to_params(row: pd.Series) -> TagParams:
    # Obtención robusta de columnas con defaults si no existen.
    def gv(col: str, default: float = 0.0) -> float:
        return get_num(row[col]) if col in row else default

    return TagParams(
        TAG=str(row.get("TAG", "SIN_TAG")),
        J_m=gv("J_m"),
        J_driver=gv("J_driver"),
        J_driven=gv("J_driven"),
        J_imp=gv("J_imp"),
        n_motor_nom=max(gv("n_motor_nom", 1500.0), 1e-6),
        n_bomba_nom=max(gv("n_bomba_nom", 1000.0), 1e-6),
        P_motor_kw=max(gv("P_motor_kw", 75.0), 0.0),
        T_disp=gv("T_disp"),  # si 0 -> estimar
        Q_nom_m3h=max(gv("Q_nom_m3h", 200.0), 1e-9),
        H0_m=gv("H0_m", 25.0),
        K=gv("K", 0.002),
        eta_h=min(max(gv("eta_h", 0.70), 0.05), 0.95),
        rho_kgm3=max(gv("rho_kgm3", 1000.0), 1.0),
        Q_min_m3h=max(gv("Q_min_m3h", 10.0), 0.0),
        Q_max_m3h=max(gv("Q_max_m3h", 600.0), 0.0),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Entradas")
    up = st.file_uploader("Subir dataset (.xlsx o .csv)", type=["xlsx", "csv"])
    df = load_dataset(up)

    if "TAG" not in df.columns:
        st.error("El dataset debe tener una columna `TAG`.")
        st.stop()

    tag_list = sorted(df["TAG"].astype(str).unique().tolist())
    sel_tag = st.selectbox("TAG", options=tag_list, index=0)

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        n_ini = st.number_input("n inicial bomba [rpm]", value=0.0, min_value=0.0, step=10.0)
    with col_r2:
        n_fin = st.number_input("n objetivo bomba [rpm]", value=1000.0, min_value=1.0, step=10.0)

    rampa_vdf = st.slider("Rampa VDF [rpm/s] (motor)", min_value=1, max_value=500, value=200, step=5)

    t_max = st.slider("t_max integración [s]", min_value=1, max_value=120, value=30, step=1)
    dt = st.select_slider("dt [s]", options=[0.001, 0.002, 0.005, 0.01, 0.02], value=0.01)


# ──────────────────────────────────────────────────────────────────────────────
# Parámetros del TAG seleccionado
# ──────────────────────────────────────────────────────────────────────────────
row = df[df["TAG"].astype(str) == str(sel_tag)].iloc[0]
params = row_to_params(row)

st.subheader(f"📌 Parámetros – **{params.TAG}**")

# Ecuación J_eq
st.latex(r"J_{\mathrm{eq}} = J_m + J_{\mathrm{driver}} + \dfrac{J_{\mathrm{driven}}+J_{\mathrm{imp}}}{r^2}")
st.caption(r"$r = n_{\mathrm{motor}}/n_{\mathrm{bomba}}$; las inercias del lado bomba giran a $\omega_p=\omega_m/r$.")

# Badges (formateo correcto del H0_m como string)
c1, c2, c3, c4 = st.columns(4)
with c1:
    show_badge(f"{params.J_eq:.3f}", "kg·m²", "J_eq")
    show_badge(f"{params.r:.2f}", "", "Relación r")
with c2:
    show_badge(f"{params.n_motor_nom:.0f}", "rpm", "n_motor_nom")
    show_badge(f"{params.n_bomba_nom:.0f}", "rpm", "n_bomba_nom")
with c3:
    show_badge(f"{params.Q_nom_m3h:.1f}", "m³/h", "Q_nom")
    show_badge(f"{params.H0_m:.2f}", "m", "H0")
with c4:
    show_badge(f"{params.K:.4f}", "–", "K")
    show_badge(f"{params.eta_h:.2f}", "–", "η")

# ──────────────────────────────────────────────────────────────────────────────
# Métricas inerciales (sin hidráulica): t_par y t_rampa
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("### ⚡ Estimación inercial vs rampa VDF")

# Estimar T_disp si es 0: T = 9550 * P[kW] / n[rpm] a nominal motor
T_disp = params.T_disp
if T_disp <= 0.0:
    T_disp = 9550.0 * max(params.P_motor_kw, 1e-9) / max(params.n_motor_nom, 1e-6)

omega_dot_torque = (T_disp / max(params.J_eq, 1e-9))  # [rad/s²] sobre el eje motor
# Pasar a \dot n motor [rpm/s]: \dot n = (60/2π)·\dot ω
n_dot_torque_motor = (60.0 / (2.0 * np.pi)) * omega_dot_torque

# Δn motor equivalente del salto en n bomba: Δn_motor = r * Δn_bomba
delta_n_bomba = max(n_fin - n_ini, 0.0)
delta_n_motor = params.r * delta_n_bomba

t_par = delta_n_motor / max(n_dot_torque_motor, 1e-9)         # limitado por par
t_rampa = delta_n_motor / max(float(rampa_vdf), 1e-9)          # limitado por rampa VDF
t_inercial_aprox = max(t_par, t_rampa)

mcol1, mcol2, mcol3 = st.columns(3)
with mcol1:
    st.metric("t_par (solo par)", f"{t_par:.2f} s")
with mcol2:
    st.metric("t_rampa (VDF)", f"{t_rampa:.2f} s")
with mcol3:
    st.metric("t_reacción aprox", f"{t_inercial_aprox:.2f} s")

st.caption(
    r"Ecuaciones: $\dot n_{\mathrm{torque}}=\frac{60}{2\pi}\frac{T_{\mathrm{disp}}}{J_{\mathrm{eq}}}$, "
    r"$t_{\mathrm{par}}=\frac{\Delta n_{\mathrm{motor}}}{\dot n_{\mathrm{torque}}}$, "
    r"$t_{\mathrm{rampa}}=\frac{\Delta n_{\mathrm{motor}}}{\mathrm{rampa}_{\mathrm{VDF}}}$."
)

# ──────────────────────────────────────────────────────────────────────────────
# Dinámica con hidráulica (modelo simple)
# J_eq·ω̇_m = T_disp − T_pump/r
# con T_pump = ρgQH(Q)/(η ω_p), ω_p=ω_m/r, Q ∝ n_p (afinidad), clamp [Qmin, Qmax]
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("🌊 Integración con hidráulica (modelo simple)")

st.latex(
    r"J_{eq}\,\dot\omega_m = T_{disp} - \dfrac{T_{pump}}{r}, \qquad "
    r"T_{pump}=\dfrac{\rho\,g\,Q\,H(Q)}{\eta\,\omega_p}, \; \omega_p=\omega_m/r, \; Q\propto n_p."
)
st.caption(
    r"Se impone además un límite de rampa VDF en el motor: $n_m(t)$ no puede crecer "
    r"más rápido que $\mathrm{rampa}_{\mathrm{VDF}}$."
)

g = 9.81

def H_of_Q(Q_m3h: float, H0: float, K: float) -> float:
    """Curva H(Q) = H0 + K*(Q/3600)^2 con Q en m³/h, devuelve H en m."""
    return H0 + K * (Q_m3h / 3600.0) ** 2

def simulate(params: TagParams,
             n_ini_bomba: float,
             n_fin_bomba: float,
             rampa_vdf_motor: float,
             t_max: float,
             dt: float,
             T_disp_const: float) -> Dict[str, np.ndarray]:
    steps = int(np.ceil(t_max / dt))
    t = np.zeros(steps + 1)
    n_b = np.zeros(steps + 1)      # rpm bomba actual
    n_m = np.zeros(steps + 1)      # rpm motor actual
    n_cmd_m = np.zeros(steps + 1)  # rpm motor comandada por VDF
    Q = np.zeros(steps + 1)        # m³/h
    PkW = np.zeros(steps + 1)      # kW hidráulica

    n_b[0] = max(n_ini_bomba, 0.0)
    n_m[0] = params.r * n_b[0]
    n_cmd_m[0] = n_m[0]

    n_fin_m = params.r * max(n_fin_bomba, 1.0)

    for k in range(steps):
        t[k + 1] = t[k] + dt

        # VDF: referencia de velocidad motor por rampa
        inc = rampa_vdf_motor * dt
        n_cmd_m[k + 1] = min(n_cmd_m[k] + inc, n_fin_m)

        # Estado actual
        omega_m = 2.0 * np.pi * n_m[k] / 60.0
        omega_p = max(omega_m / max(params.r, 1e-9), 1e-6)

        # Afinidad Q ~ n_bomba
        n_b[k] = max(n_b[k], 0.0)
        Q_k = params.Q_nom_m3h * (n_b[k] / max(params.n_bomba_nom, 1e-6))
        # Clamp hidráulico
        Q_k = min(max(Q_k, params.Q_min_m3h), params.Q_max_m3h)

        # Curva de bomba
        H = H_of_Q(Q_k, params.H0_m, params.K)

        # Par de bomba (lado bomba): T_pump = ρ g Q H / (η ω_p)
        Q_m3s = Q_k / 3600.0
        T_pump = (params.rho_kgm3 * g * Q_m3s * H) / max(params.eta_h * omega_p, 1e-6)

        # Dinámica en el eje motor
        domega_m = (T_disp_const - (T_pump / max(params.r, 1e-9))) / max(params.J_eq, 1e-9)
        omega_m_new = max(omega_m + domega_m * dt, 0.0)
        n_m_candidate = 60.0 * omega_m_new / (2.0 * np.pi)

        # Limitación por rampa (nunca superar n_cmd_m)
        n_m[k + 1] = min(n_m_candidate, n_cmd_m[k + 1])
        n_b[k + 1] = n_m[k + 1] / max(params.r, 1e-9)

        # Potencia hidráulica
        Q_k_next = params.Q_nom_m3h * (n_b[k + 1] / max(params.n_bomba_nom, 1e-6))
        Q_k_next = min(max(Q_k_next, params.Q_min_m3h), params.Q_max_m3h)
        H_next = H_of_Q(Q_k_next, params.H0_m, params.K)
        PkW[k + 1] = (params.rho_kgm3 * g * (Q_k_next / 3600.0) * H_next) / 1000.0

        Q[k + 1] = Q_k_next

        # Si ya alcanzamos n_fin_m (tolerancia), podemos cortar
        if n_m[k + 1] >= n_fin_m - 1e-6 and n_cmd_m[k + 1] >= n_fin_m - 1e-6:
            # recortar arrays
            last = k + 1
            t = t[: last + 1]
            n_b = n_b[: last + 1]
            n_m = n_m[: last + 1]
            n_cmd_m = n_cmd_m[: last + 1]
            Q = Q[: last + 1]
            PkW = PkW[: last + 1]
            break

    return {"t": t, "n_b": n_b, "n_m": n_m, "n_cmd_m": n_cmd_m, "Q": Q, "PkW": PkW}


sim = simulate(
    params=params,
    n_ini_bomba=n_ini,
    n_fin_bomba=n_fin,
    rampa_vdf_motor=float(rampa_vdf),
    t_max=float(t_max),
    dt=float(dt),
    T_disp_const=float(T_disp),
)

# ──────────────────────────────────────────────────────────────────────────────
# Gráfico Plotly con 3 ejes corregidos (sin error de position fuera de [0,1])
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("#### 📈 Curvas de arranque")

if go is None:
    st.warning("Plotly no está disponible. Se omitirá el gráfico interactivo.")
else:
    fig = go.Figure()
    t = sim["t"]
    Q = sim["Q"]
    n_p = sim["n_b"]
    P = sim["PkW"]

    fig.add_trace(go.Scatter(x=t, y=Q, name="Q [m³/h]", yaxis="y"))
    fig.add_trace(go.Scatter(x=t, y=n_p, name="n bomba [rpm]", yaxis="y2"))
    fig.add_trace(go.Scatter(x=t, y=P, name="P hidráulica [kW]", yaxis="y3"))
    fig.add_trace(go.Scatter(x=t, y=sim["n_cmd_m"] / max(params.r, 1e-9),
                             name="n bomba cmd [rpm]", yaxis="y2",
                             line=dict(dash="dash")))

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        xaxis=dict(title="Tiempo [s]"),

        # Eje izquierdo (Q)
        yaxis=dict(title="Q [m³/h]", anchor="x"),

        # Eje derecho (n)
        yaxis2=dict(title="n bomba [rpm]", overlaying="y", side="right", anchor="x"),

        # Tercer eje a la derecha con anchor free y position < 1 (FIX)
        yaxis3=dict(title="P [kW]", overlaying="y", side="right", anchor="free", position=0.98),
    )
    st.plotly_chart(fig, use_container_width=True)

st.caption(
    r"Ecuaciones: $J_{eq}=J_m+J_{driver}+(J_{driven}+J_{imp})/r^2$, "
    r"$T_{pump}=\frac{\rho g Q H(Q)}{\eta\,\omega_p}$, con $Q\propto n_p$ y "
    r"$H(Q)=H_0+K\left(\frac{Q}{3600}\right)^2$."
)

# ──────────────────────────────────────────────────────────────────────────────
# Reportes y descargas (por BYTES, sin rutas -> evita MediaFileHandler)
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("📥 Reportes")

# Reporte del TAG actual (resumen + última simulación)
rep_tag = pd.DataFrame(
    {
        "TAG": [params.TAG],
        "J_eq_kgm2": [params.J_eq],
        "r": [params.r],
        "T_disp_Nm": [T_disp],
        "n_ini_bomba_rpm": [n_ini],
        "n_fin_bomba_rpm": [n_fin],
        "rampa_vdf_motor_rpmps": [rampa_vdf],
        "t_par_s": [t_par],
        "t_rampa_s": [t_rampa],
        "t_reaccion_aprox_s": [max(t_par, t_rampa)],
    }
)

csv_one = csv_bytes_from_df(rep_tag)

st.download_button(
    "⬇️ Descargar reporte del TAG seleccionado",
    data=csv_one,
    file_name=f"reporte_{params.TAG}_rampa_{int(rampa_vdf)}rpmps.csv",
    mime="text/csv",
)

# Reporte para TODOS los TAG con la rampa actual
def build_all_tags_report(df: pd.DataFrame, rampa: float) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        p = row_to_params(r)
        # Estimar T_disp si no hay
        T = p.T_disp if p.T_disp > 0 else 9550.0 * max(p.P_motor_kw, 1e-9) / max(p.n_motor_nom, 1e-6)
        omega_dot = T / max(p.J_eq, 1e-9)
        n_dot = (60.0 / (2.0 * np.pi)) * omega_dot
        # Δn_motor equivalente para 0 -> n_bomba_nom (referencia común)
        dnm = p.r * max(p.n_bomba_nom, 1.0)
        t_par_i = dnm / max(n_dot, 1e-9)
        t_rampa_i = dnm / max(rampa, 1e-9)
        rows.append(
            {
                "TAG": p.TAG,
                "J_eq_kgm2": p.J_eq,
                "r": p.r,
                "T_disp_Nm": T,
                "n_obj_bomba_rpm": p.n_bomba_nom,
                "rampa_vdf_motor_rpmps": rampa,
                "t_par_s": t_par_i,
                "t_rampa_s": t_rampa_i,
                "t_reaccion_aprox_s": max(t_par_i, t_rampa_i),
                "H0_m": p.H0_m,
                "K": p.K,
                "Q_nom_m3h": p.Q_nom_m3h,
            }
        )
    return pd.DataFrame(rows)

rep_all = build_all_tags_report(df, float(rampa_vdf))
csv_all = csv_bytes_from_df(rep_all)

st.download_button(
    "⬇️ Descargar reporte (todos los TAG, con rampa seleccionada)",
    data=csv_all,
    file_name=f"reporte_todos_los_TAG_rampa_{int(rampa_vdf)}rpmps.csv",
    mime="text/csv",
)

st.markdown("---")
st.markdown("#### 📝 Notas")
st.markdown(
    r"""
- El modelo hidráulico es intencionalmente simple y usa $Q\propto n_p$ y $H(Q)=H_0+K\left(\frac{Q}{3600}\right)^2$.
- Si la lectura del dataset trae celdas con texto (comas, unidades), se parsean con una rutina robusta.
- Los botones de descarga generan **bytes en memoria** (no rutas), evitando errores `MediaFileHandler`.
- El gráfico Plotly usa un **tercer eje** en `position=0.98` (dentro de [0,1]) para evitar el error de `layout.yaxis.position`.
"""
)
