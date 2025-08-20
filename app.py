# app.py
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st

# =========================
# Utilidades de formato
# =========================
def f2(x: float | int | None) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def latex_num(x: float) -> str:
    # para fórmulas; usa punto decimal
    return f"{x:.2f}"

def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())

def pick(df: pd.DataFrame, names: Tuple[str, ...], default=None):
    pool = {norm(c): c for c in df.columns}
    for n in names:
        key = norm(n)
        if key in pool:
            return pool[key]
    return default

# =========================
# Carga robusta del dataset
# =========================
DATA_FILE_NAME = "bombas_dataset_with_torque_params.xlsx"

def _resolve_data_path() -> Path | None:
    here = Path(__file__).resolve().parent
    for p in [here / DATA_FILE_NAME, Path.cwd() / DATA_FILE_NAME]:
        if p.exists():
            return p
    # búsqueda recursiva
    for p in here.rglob("*.xlsx"):
        if p.name.lower() == DATA_FILE_NAME.lower():
            return p
    return None

def _looks_like_tag_series(s: pd.Series) -> int:
    try:
        return s.astype(str).str.contains(r"\d{3,5}-PU-\d{3}", flags=re.I, regex=True).sum()
    except Exception:
        return 0

def _pick_best_sheet(xls: pd.ExcelFile) -> str:
    best_name, best_score = xls.sheet_names[0], -1
    for name in xls.sheet_names:
        dfh = xls.parse(name, nrows=100)
        score = 0
        if not dfh.empty:
            score += _looks_like_tag_series(dfh.iloc[:, 0])
            score += dfh.shape[1]
        if score > best_score:
            best_name, best_score = name, score
    return best_name

@st.cache_data(show_spinner=True)
def load_dataset() -> Tuple[pd.DataFrame, str, Path]:
    data_path = _resolve_data_path()
    if data_path is None:
        raise FileNotFoundError(f"No encontré {DATA_FILE_NAME} junto a app.py ni en el directorio actual.")
    xls = pd.ExcelFile(data_path, engine="openpyxl")
    lower = [s.lower() for s in xls.sheet_names]
    preferred = None
    if "dataset" in lower:
        preferred = xls.sheet_names[lower.index("dataset")]
    else:
        cands = [s for s in xls.sheet_names if re.search(r"(data\s*set|dataset|data)", s, re.I)]
        preferred = cands[0] if cands else None
    sheet = preferred or _pick_best_sheet(xls)
    df = xls.parse(sheet)
    df = df.dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]
    return df, sheet, data_path

df, used_sheet, used_path = load_dataset()
st.caption(f"Archivo: **{used_path.name}** · Hoja: **{used_sheet}** · {df.shape[0]} filas × {df.shape[1]} columnas")

# Por convenio: 1ª columna = TAG (único)
COL_TAG = df.columns[0]
TAGS = df[COL_TAG].astype(str).tolist()

# ===========
# Mapeo suave
# ===========
COL_RATIO = pick(df, ("RelacionTransmision", "Ratio", "Relación", "B", "reltransmision", "r"))
COL_PMOT = pick(df, ("MotorPowerInstalled_kW", "MotorPower_kW", "P_motor_kW"))
COL_JM   = pick(df, ("Motor_J_kgm2", "J_m", "Jmotor"))
COL_JDRV = pick(df, ("J_driver_total_kgm2", "J_driver_kgm2", "Jdriver"))
COL_JDRN = pick(df, ("J_driven_total_kgm2", "J_driven_kgm2", "Jdriven"))
COL_JIMP = pick(df, ("Impeller_J_kgm2", "J_imp_kgm2", "Jimp"))
COL_NM_MIN = pick(df, ("MotorSpeedMin_rpm","n_m_min_rpm","n_min_motor"))
COL_NM_MAX = pick(df, ("MotorSpeedMax_rpm","n_m_max_rpm","n_max_motor"))
COL_NP_MIN = pick(df, ("PumpSpeedMin_rpm","n_p_min_rpm","n_min_bomba"))
COL_NP_MAX = pick(df, ("PumpSpeedMax_rpm","n_p_max_rpm","n_max_bomba"))
# Hidráulica
COL_H0    = pick(df, ("H0_m","H0"))
COL_K     = pick(df, ("K_m_s2","K"))
COL_RHO   = pick(df, ("rho_kgm3","rho"))
COL_QMIN  = pick(df, ("Q_min_m3h","Qmin"))
COL_QMAX  = pick(df, ("Q_max_m3h","Qmax"))
COL_ETA_A = pick(df, ("eta_a",))
COL_ETA_B = pick(df, ("eta_b",))
COL_ETA_C = pick(df, ("eta_c",))

# ===========
# Selección
# ===========
st.sidebar.header("Selección")
tag = st.sidebar.selectbox("TAG", TAGS, index=0)
row = df.loc[df[COL_TAG].astype(str) == tag].iloc[0]

# ===========
# Lectura de datos por TAG con fallback
# ===========
def get_float(r, col, default=0.0):
    try:
        val = float(r[col])
        if np.isnan(val): return default
        return val
    except Exception:
        return default

r = get_float(row, COL_RATIO, 1.0) if COL_RATIO else 1.0  # n_p = n_m / r

P_kW   = get_float(row, COL_PMOT, 0.0)
J_m    = get_float(row, COL_JM,   0.0)
J_drv  = get_float(row, COL_JDRV, 0.0)
J_drn  = get_float(row, COL_JDRN, 0.0)
J_imp  = get_float(row, COL_JIMP, 0.0)

n_m_min = get_float(row, COL_NM_MIN, 0.0)
n_m_max = get_float(row, COL_NM_MAX, 0.0)

# Si no vienen n_p_* usamos la relación r con n_m_*
if COL_NP_MIN and COL_NP_MAX:
    n_p_min = get_float(row, COL_NP_MIN, max(1.0, n_m_min / r if r else 1.0))
    n_p_max = get_float(row, COL_NP_MAX, max(1.0, n_m_max / r if r else 1.0))
else:
    n_p_min = max(1.0, n_m_min / r if r else 1.0)
    n_p_max = max(1.0, n_m_max / r if r else 1.0)

# Hidráulica
H0   = get_float(row, COL_H0, 0.0)
K    = get_float(row, COL_K, 0.0)
rho  = get_float(row, COL_RHO, 1000.0)
Qmin = get_float(row, COL_QMIN, 0.0)      # m3/h
Qmax = get_float(row, COL_QMAX, 0.0)
eta_a = get_float(row, COL_ETA_A, 0.0)
eta_b = get_float(row, COL_ETA_B, 0.0)
eta_c = get_float(row, COL_ETA_C, 0.0)

# Torque nominal estimado si no está en dataset:
# T ≈ 9550 * P[kW] / n[rpm] (suponemos n ≈ n_m_max en zona de par constante)
T_nom_est = 9550.0 * P_kW / (n_m_max if n_m_max > 0 else 1500.0)

# =================
# Sección 1 - Ficha
# =================
st.markdown("# Memoria de cálculo – Tiempo de reacción de bombas (VDF)")
st.markdown("## 1) Datos del equipo seleccionado")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.write("**TAG**"); st.write(f"{tag}")
with col2:
    st.write("**Modelo motor**"); st.write(f"{f2(P_kW)} kW")
with col3:
    st.write("**Relación transmisión (r)**"); st.write(f"{f2(r)}")
with col4:
    st.write("**Impulsor J**"); st.write(f"{f2(J_imp)} kg·m²")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.write("**J_motor (J_m)**"); st.write(f"{f2(J_m)} kg·m²")
with col2:
    st.write("**J_driver (polea+manguito)**"); st.write(f"{f2(J_drv)} kg·m²")
with col3:
    st.write("**J_driven (polea+manguito)**"); st.write(f"{f2(J_drn)} kg·m²")
with col4:
    st.write("**n_motor min–max [rpm]**"); st.write(f"{f2(n_m_min)} – {f2(n_m_max)}")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.write("**n_bomba min–max [rpm]**"); st.write(f"{f2(n_p_min)} – {f2(n_p_max)}")
with col2:
    st.write("**H(Q) = H₀ + K·Q²**"); st.write(f"H₀={f2(H0)} m, K={f2(K)} m·s²/m⁶")
with col3:
    st.write("**η(Q) = ηₐ + η_b·Q + η_c·Q²**"); st.write(f"ηₐ={f2(eta_a)}, η_b={f2(eta_b)}, η_c={f2(eta_c)}")
with col4:
    st.write("**ρ (slurry)**"); st.write(f"{f2(rho)} kg/m³")

st.divider()

# =========================================
# Sección 2 – Dinámica inercial (sin hidráulica)
# =========================================
st.markdown("## 2) Dinámica inercial (sin efectos hidráulicos)")

J_eq = J_m + J_drv + (J_drn + J_imp) / (r**2 if r > 0 else 1.0)

# Entradas de cálculo (compactas)
ci, cf, ct = st.columns(3)
with ci:
    n_i = st.number_input("Velocidad Motor inicial [rpm]", value=float(max(n_m_min, 0.0)), step=1.0, format="%.2f")
with cf:
    n_f = st.number_input("Velocidad Motor final [rpm]", value=float(max(n_m_min, min(n_m_max, n_m_min + 500))), step=1.0, format="%.2f")
with ct:
    T_disp = st.number_input("Par disponible T_disp [Nm]", value=float(max(10.0, T_nom_est)), step=1.0, format="%.2f")

rampa_vdf = st.slider("Rampa VDF [rpm/s] (motor)", min_value=50, max_value=800, value=300, step=10)

st.markdown("**Fórmulas**")
st.latex(r"J_{eq} = J_m + J_{driver} + \frac{J_{driven} + J_{imp}}{r^2}")
st.latex(r"\dot n_{\mathrm{torque}} = \frac{60}{2\pi}\,\frac{T_{\mathrm{disp}}}{J_{eq}}")
st.latex(r"t_{\mathrm{par}} = \frac{\Delta n}{\dot n_{\mathrm{torque}}} \qquad t_{\mathrm{rampa}} = \frac{\Delta n}{\mathrm{rampa}_{\mathrm{VDF}}} \qquad t_{\mathrm{final}}=\max\{t_{\mathrm{par}},\,t_{\mathrm{rampa}}\}")

Delta_n = max(0.0, n_f - n_i)
n_dot_torque = (60.0/(2*np.pi)) * (T_disp / max(J_eq, 1e-9))
t_par   = Delta_n / max(n_dot_torque, 1e-9)
t_rampa = Delta_n / max(float(rampa_vdf), 1e-9)
t_final_sin = max(t_par, t_rampa)

colA, colB, colC, colD = st.columns(4)
with colA: st.metric("J_eq [kg·m²]", f2(J_eq))
with colB: st.metric("Δn [rpm]", f2(Delta_n))
with colC: st.metric(r"\dot n_{\mathrm{torque}} [rpm/s]", f2(n_dot_torque))
with colD: st.metric("t_final (sin hidráulica) [s]", f2(t_final_sin))

with st.expander("Sustitución numérica"):
    st.latex(
        r"J_{eq} = " + latex_num(J_m) + r" + " + latex_num(J_drv) + r" + \dfrac{" +
        latex_num(J_drn) + r" + " + latex_num(J_imp) + r"}{" + latex_num(r) + r"^{2}} \Rightarrow " +
        latex_num(J_eq) + r"\ \mathrm{kg\cdot m^2}"
    )
    st.latex(r"\Delta n = " + latex_num(Delta_n) + r"\ \mathrm{rpm}")
    st.latex(
        r"\dot n_{\mathrm{torque}} = \frac{60}{2\pi}\frac{" + latex_num(T_disp) + r"}{" +
        latex_num(J_eq) + r"} = " + latex_num(n_dot_torque) + r"\ \mathrm{rpm/s}"
    )
    st.latex(
        r"t_{\mathrm{par}} = \frac{" + latex_num(Delta_n) + r"}{" + latex_num(n_dot_torque) +
        r"} = " + latex_num(t_par) + r"\ \mathrm{s},\quad " +
        r"t_{\mathrm{rampa}} = \frac{" + latex_num(Delta_n) + r"}{" + latex_num(rampa_vdf) +
        r"} = " + latex_num(t_rampa) + r"\ \mathrm{s}"
    )
    st.latex(r"t_{\mathrm{final}} = \max(" + latex_num(t_par) + r"," + latex_num(t_rampa) + r") = " + latex_num(t_final_sin) + r"\ \mathrm{s}")

st.divider()

# ===================================================
# Sección 3 – Dinámica con carga hidráulica (integrada)
# ===================================================
st.markdown("## 3) Dinámica con carga hidráulica")

st.markdown("**Modelo**")
st.latex(r"H(Q)=H_0 + K\,Q^2 \qquad \eta(Q)=\eta_a + \eta_b Q + \eta_c Q^2")
st.latex(r"P_h = \rho g\,Q\,H(Q) \qquad T_{\mathrm{load},p}=\dfrac{P_h}{\omega_p},\quad \omega_p=\dfrac{2\pi n_p}{60}")
st.latex(r"T_{\mathrm{load},m} = \dfrac{T_{\mathrm{load},p}}{r} \qquad J_{eq}\,\dot\omega_m = T_{\mathrm{disp}} - T_{\mathrm{load},m}")
st.caption("Para Q(n_p) se interpola linealmente entre (n_p_min, Q_min) y (n_p_max, Q_max). Integración explícita con tope de rampa del VDF en el eje del motor.")

# ---- Rango de rpm de bomba: slider seguro (enteros coherentes) ----
n_p_lo = int(max(1, round(min(n_p_min, n_p_max))))
n_p_hi = int(max(n_p_lo + 1, round(max(n_p_min, n_p_max))))

span = max(1, n_p_hi - n_p_lo)
default_lo = int(n_p_lo + 0.1 * span)
default_hi = int(n_p_hi - 0.1 * span)
if default_hi <= default_lo:
    default_lo = n_p_lo
    default_hi = n_p_hi

n1_p, n2_p = st.slider(
    "Rango de velocidad de bomba [rpm]",
    min_value=n_p_lo,
    max_value=n_p_hi,
    value=(default_lo, default_hi),
    step=1,
)

# Convertimos a motor usando r (n_m = r * n_p)
n1_m = r * n1_p
n2_m = r * n2_p

# Interpolador de caudal en m3/h -> m3/s
def Q_of_np(np_rpm: float) -> float:
    if n_p_hi == n_p_lo or (Qmax - Qmin) == 0:
        q_m3h = Qmin
    else:
        frac = (np_rpm - n_p_lo) / (n_p_hi - n_p_lo)
        frac = max(0.0, min(1.0, frac))
        q_m3h = Qmin + frac * (Qmax - Qmin)
    return q_m3h / 3600.0

def eta_of_Q(Q_m3s: float) -> float:
    return max(1e-3, min(0.98, eta_a + eta_b*Q_m3s*3600.0 + eta_c*(Q_m3s*3600.0)**2))

g = 9.81

def T_load_motor(n_p: float) -> float:
    # torque hidráulico equivalente en eje del motor (Nm)
    Q = Q_of_np(n_p)
    H = H0 + K * (Q**2)
    eta = eta_of_Q(Q)
    P_h = rho * g * Q * H / max(eta, 1e-6)  # W
    w_p = 2*np.pi * n_p / 60.0
    T_load_p = P_h / max(w_p, 1e-6)
    return T_load_p / max(r, 1e-6)

def integrate_with_load(n_m_ini: float, n_m_fin: float, dt: float = 0.02) -> Tuple[float, pd.DataFrame]:
    # integración hacia arriba o abajo
    sign = 1.0 if n_m_fin >= n_m_ini else -1.0
    n_m = n_m_ini
    t = 0.0
    rows = []
    while (sign > 0 and n_m < n_m_fin - 1e-6) or (sign < 0 and n_m > n_m_fin + 1e-6):
        n_p = n_m / max(r, 1e-9)
        Tload_m = T_load_motor(n_p)
        # aceleración por par
        n_dot_par = (60.0/(2*np.pi)) * (T_disp - Tload_m) / max(J_eq, 1e-9)  # rpm/s (motor)
        # límite de rampa VDF
        n_dot_lim = rampa_vdf if sign > 0 else -rampa_vdf
        n_dot = np.clip(n_dot_par, -abs(rampa_vdf), abs(rampa_vdf))
        # paso
        n_new = n_m + n_dot * dt
        # evitar sobrepasar
        if sign > 0:
            n_new = min(n_new, n_m_fin)
        else:
            n_new = max(n_new, n_m_fin)
        rows.append({"t": t, "n_m": n_m, "n_p": n_p, "Q_m3s": Q_of_np(n_p), "T_load_m": Tload_m, "n_dot": n_dot})
        n_m = n_new
        t += dt
        # condición de bloqueo si el par no alcanza (aceleración ~ 0 por 1 s)
        if abs(n_dot) < 1e-6 and t > 1.0:
            break
    df_path = pd.DataFrame(rows)
    return t, df_path

t_hid, df_path = integrate_with_load(n1_m, n2_m, dt=0.02)

colX, colY, colZ = st.columns(3)
with colX: st.metric("Velocidad Bomba (inicio–fin) [rpm]", f"{int(n1_p)} – {int(n2_p)}")
with colY: st.metric("Δn_motor [rpm]", f2(abs(n2_m - n1_m)))
with colZ: st.metric("Tiempo con hidráulica [s]", f2(t_hid))

st.caption("Si la aceleración por par disponible es menor que el tope de rampa del VDF, domina la hidráulica; de lo contrario, domina la rampa.")

# Exportable
def build_summary_row() -> Dict[str, float]:
    return {
        "TAG": tag,
        "r": r,
        "J_eq_kgm2": round(J_eq, 6),
        "n_m_ini_rpm": round(n1_m, 2),
        "n_m_fin_rpm": round(n2_m, 2),
        "n_p_ini_rpm": int(n1_p),
        "n_p_fin_rpm": int(n2_p),
        "Delta_n_motor_rpm": round(abs(n2_m - n1_m), 2),
        "rampa_VDF_rpms": float(rampa_vdf),
        "T_disp_Nm": round(T_disp, 3),
        "t_final_sin_hid_s": round(t_final_sin, 3),
        "t_final_con_hid_s": round(t_hid, 3),
    }

summary_df = pd.DataFrame([build_summary_row()])
st.dataframe(summary_df, use_container_width=True)
st.download_button(
    "Descargar resumen (CSV)",
    data=summary_df.to_csv(index=False).encode("utf-8"),
    file_name=f"resumen_{tag}.csv",
    mime="text/csv",
)
