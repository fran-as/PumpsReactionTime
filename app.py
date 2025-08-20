# app.py
import math
import re
from io import StringIO
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------
# Utilidades
# ------------------------------
def to_float(x) -> float:
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

def fmt2(x, unit=""):
    try:
        v = float(x)
    except Exception:
        return str(x)
    s = f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{s} {unit}".strip()

def get_col(df: pd.DataFrame, *pats, default=None) -> Optional[str]:
    rx = [re.compile(p, re.I) for p in pats]
    for c in df.columns:
        if all(r.search(str(c)) for r in rx):
            return c
    return default

# ------------------------------
# App config
# ------------------------------
st.set_page_config(page_title="Memoria de Cálculo – Tiempo de reacción de bombas (VDF)", layout="wide")
st.title("Memoria de Cálculo – Tiempo de reacción de bombas (VDF)")

DATA_FILE = "bombas_dataset_with_torque_params.xlsx"

@st.cache_data(show_spinner=True)
def load_data(path: str) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    sheet_name = "dataset" if "dataset" in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    return df

try:
    df = load_data(DATA_FILE)
except Exception as e:
    st.error(f"No pude abrir **{DATA_FILE}**. Verifica que esté en la raíz. Detalle: {e}")
    st.stop()

# Por convenio: Columna A = TAG (único), Columna B = relación r_trans
COL_TAG = df.columns[0]
COL_R   = df.columns[1]  # segunda columna SIEMPRE relación transmisión
TAGS = df[COL_TAG].astype(str).tolist()

# ------------------------------
# Lectura de fila seleccionada
# ------------------------------
st.sidebar.header("Selección")
tag = st.sidebar.selectbox("TAG", TAGS, index=0)

row = df.loc[df[COL_TAG].astype(str) == tag]
if row.empty:
    st.error("TAG no encontrado en el dataset.")
    st.stop()
row = row.iloc[0]

# ------------------------------
# Carga de parámetros desde la fila
# ------------------------------
def read_inputs(rw: pd.Series) -> Dict[str, float]:
    # r: columna B
    r_tr = max(to_float(rw[COL_R]), 1e-9)

    # Motor
    n_min = to_float(rw.get(get_col(df, r"motor", r"min", r"rpm"), 0.0))
    n_max = to_float(rw.get(get_col(df, r"motor", r"max", r"rpm"), 0.0))
    n_nom = to_float(rw.get(get_col(df, r"n_ref_rpm"), n_max if n_max > 0 else 1500.0))
    T_nom = to_float(rw.get(get_col(df, r"T.*nom.*Nm"), 0.0))
    if T_nom <= 0:
        P_kw = to_float(rw.get(get_col(df, r"(MotorPower|motor_power|Power|Potenc)", r"kW"), 0.0))
        # T [Nm] = 9550 * P[kW] / n[rpm]
        T_nom = 9550.0 * P_kw / max(n_nom if n_nom > 0 else (n_max if n_max > 0 else 1500.0), 1e-9)

    # Inercias (si no están, 0)
    J_m   = to_float(rw.get(get_col(df, r"J", r"motor", r"kgm"), 0.0))
    J_imp = to_float(rw.get(get_col(df, r"J", r"(impeller|impulsor)", r"kgm"), 0.0))
    J_drv = to_float(rw.get(get_col(df, r"J", r"driver", r"pulley|polea", r"kgm"), 0.0)) \
          + to_float(rw.get(get_col(df, r"J", r"driver", r"(bush|mangui)", r"kgm"), 0.0))
    J_drn = to_float(rw.get(get_col(df, r"J", r"driven", r"pulley|polea", r"kgm"), 0.0)) \
          + to_float(rw.get(get_col(df, r"J", r"driven", r"(bush|mangui)", r"kgm"), 0.0))

    # Bomba
    D_imp = to_float(rw.get(get_col(df, r"(D|Diam).*imp", r"mm"), 0.0))

    # Sistema (H–Q y eficiencia) – columnas fijas entregadas
    H0     = to_float(rw.get("H0_m", 0.0))
    K      = to_float(rw.get("K_m_s2", 0.0))
    rho    = to_float(rw.get("rho_kgm3", 1000.0))
    Qmin   = to_float(rw.get("Q_min_m3h", 0.0))
    Qmax   = to_float(rw.get("Q_max_m3h", 0.0))
    Qref   = to_float(rw.get("Q_ref_m3h", 0.5*(Qmin+Qmax) if (Qmax>Qmin) else 1.0))
    # Eficiencia
    eta_a  = to_float(rw.get("eta_a", 0.7))
    eta_b  = to_float(rw.get("eta_b", 0.0))
    eta_c  = to_float(rw.get("eta_c", 0.0))
    eta_beta = to_float(rw.get("eta_beta", 0.0))  # opcional
    eta_min_clip = to_float(rw.get("eta_min_clip", 0.30))
    eta_max_clip = to_float(rw.get("eta_max_clip", 0.90))

    return dict(
        r_tr=r_tr, n_min=n_min, n_max=n_max, n_nom=n_nom, T_nom=T_nom,
        J_m=J_m, J_imp=J_imp, J_drv=J_drv, J_drn=J_drn, D_imp=D_imp,
        H0=H0, K=K, rho=rho, Qmin=Qmin, Qmax=Qmax, Qref=Qref,
        eta_a=eta_a, eta_b=eta_b, eta_c=eta_c,
        eta_beta=eta_beta, eta_min_clip=eta_min_clip, eta_max_clip=eta_max_clip
    )

vals = read_inputs(row)

# ------------------------------
# Sección 1 – Parámetros de entrada y fórmulas
# ------------------------------
st.subheader("1) Parámetros de entrada")

cM, cT, cB, cS = st.columns(4)

with cM:
    st.markdown("### Motor")
    st.markdown(f"- **T_nom [Nm]**: {fmt2(vals['T_nom'])}")
    st.markdown(f"- **Velocidad Motor min–max [rpm]**: {fmt2(vals['n_min'],'rpm')} – {fmt2(vals['n_max'],'rpm')}")
    st.markdown(f"- **J_m (kg·m²)**: {fmt2(vals['J_m'])}")

with cT:
    st.markdown("### Transmisión")
    st.markdown(rf"- **Relación \(r = n_{{motor}}/n_{{bomba}}\)**: {fmt2(vals['r_tr'])}")
    st.markdown(f"- **J_driver total (kg·m²)**: {fmt2(vals['J_drv'])}")
    st.markdown(f"- **J_driven total (kg·m²)**: {fmt2(vals['J_drn'])}")

with cB:
    st.markdown("### Bomba")
    st.markdown(f"- **Diámetro impulsor [mm]**: {fmt2(vals['D_imp'],'mm')}")
    st.markdown(f"- **J_imp (kg·m²)**: {fmt2(vals['J_imp'])}")
    n_b_min = vals['n_min']/vals['r_tr']
    n_b_max = vals['n_max']/vals['r_tr']
    st.markdown(f"- **Velocidad Bomba min–max [rpm]**: {fmt2(n_b_min,'rpm')} – {fmt2(n_b_max,'rpm')}")

with cS:
    st.markdown("### Sistema (H–Q, η)")
    # Fórmulas
    st.latex(r"H(Q)=H_0+K\left(\frac{Q}{3600}\right)^2,\qquad Q\ [\mathrm{m^3/h}]")
    st.latex(r"\eta(Q,n)=\mathrm{clip}\!\left[\left(\eta_a+\eta_b\frac{Q}{Q_{\mathrm{ref}}}+\eta_c\left(\frac{Q}{Q_{\mathrm{ref}}}\right)^2\right)\left(\frac{n}{n_{\mathrm{ref}}}\right)^{\eta_\beta},\ \eta_{\min},\ \eta_{\max}\right]")
    # Datos
    st.markdown(f"- **H0 [m]**: {fmt2(vals['H0'])}  \n- **K [m·s²]**: {fmt2(vals['K'])}")
    st.markdown(f"- **ρ [kg/m³]**: {fmt2(vals['rho'])}  \n- **Q rango [m³/h]**: {fmt2(vals['Qmin'])} → {fmt2(vals['Qmax'])}")
    st.markdown(f"- **Q_ref [m³/h]**: {fmt2(vals['Qref'])}  \n- **n_ref [rpm]**: {fmt2(vals['n_nom'],'rpm')}")
    st.markdown(f"- **η coef.**: a={vals['eta_a']:.3f}, b={vals['eta_b']:.3f}, c={vals['eta_c']:.3f}")
    st.markdown(f"- **η límites**: [{vals['eta_min_clip']:.2f} , {vals['eta_max_clip']:.2f}], **η_beta**={vals['eta_beta']:.2f}")

st.markdown("---")

# ------------------------------
# Sección 2 – Inercia equivalente (sin desplegables)
# ------------------------------
st.subheader("2) Inercia equivalente al eje del motor")

st.latex(r"\omega_p=\frac{\omega_m}{r}")
st.latex(r"\frac12 J_{\mathrm{eq}}\omega_m^2=\frac12 J_m\omega_m^2+\frac12 J_{\mathrm{driver}}\omega_m^2+\frac12 J_{\mathrm{driven}}\omega_p^2+\frac12 J_{\mathrm{imp}}\omega_p^2")
st.latex(r"J_{\mathrm{eq}}=J_m+J_{\mathrm{driver}}+\frac{J_{\mathrm{driven}}+J_{\mathrm{imp}}}{r^2}")

J_eq = vals["J_m"] + vals["J_drv"] + (vals["J_drn"] + vals["J_imp"])/(vals["r_tr"]**2)

st.latex(rf"""
\begin{{aligned}}
J_{{\mathrm{{eq}}}} &= {vals['J_m']:.2f} + {vals['J_drv']:.2f} + \frac{{{vals['J_drn']:.2f}+{vals['J_imp']:.2f}}}{{({vals['r_tr']:.2f})^2}}\\
&= \mathbf{{{J_eq:.2f}}}\ \mathrm{{kg\cdot m^2}}
\end{{aligned}}
""")
st.info(f"**J_eq (kg·m²):** {fmt2(J_eq)}")

st.markdown("---")

# ==============================
# 3) Respuesta inercial (sin efectos hidráulicos)
# ==============================
st.subheader("3) Respuesta inercial (sin efectos hidráulicos)")

import math

# -----------------------------
# Utilidades sobre el dataset seleccionado (vals)
# -----------------------------
def v(keys, default=0.0):
    """Intenta leer cualquiera de estas claves del dict vals; si no están, devuelve default."""
    for k in keys:
        if k in vals and vals[k] is not None and str(vals[k]) != "":
            return float(vals[k])
    return float(default)

# Relación de transmisión r = n_motor / n_bomba
r = v(["ratio","RelacionTransmision","relacion","r","TransmissionRatio"], 1.0)

# Inercias (todas en SI: kg·m²)
J_m      = v(["J_m","J_motor","Motor_J_kgm2"], 0.0)
J_drv    = v(["J_driver","J_driver_total","J_polea_motor"], 0.0)   # polea + manguito motor
J_drn    = v(["J_driven","J_driven_total","J_polea_bomba"], 0.0)   # polea + manguito bomba
J_imp    = v(["J_imp","Impeller_J_kgm2","J_impulsor"], 0.0)

# Si ya calculaste J_eq antes, úsalo; si no, compútalo aquí con r
J_eq_from_vals = J_m + J_drv + (J_drn + J_imp)/(r**2)
J_eq = float(st.session_state.get("J_eq", J_eq_from_vals))  # permite que otra sección inyecte J_eq

# Rango de velocidades del motor (desde dataset)
n_min = v(["n_min","MotorSpeedMin_rpm","n_motor_min"], 495.0)
n_max = v(["n_max","MotorSpeedMax_rpm","n_motor_max"], 990.0)
T_nom = v(["T_nom","T_motor_nominal","Par_nominal_Nm"], 240.0)

# -----------------------------------
# Definiciones (texto + símbolos)
# -----------------------------------
st.latex(r"\textbf{Definiciones}")
st.latex(r"\sum \tau = J\,\alpha \quad\Rightarrow\quad \alpha = \dfrac{\tau_{\mathrm{disp}}-\tau_{\mathrm{load}}}{J_{\mathrm{eq}}}")
st.caption("Balance de par rotacional; aquí consideraremos únicamente la inercia (sin par hidráulico).")

st.latex(r"\dot{n}_{\mathrm{torque}} = \dfrac{60}{2\pi}\,\alpha"
         r"\quad\Rightarrow\quad"
         r"\dot{n}_{\mathrm{torque}}=\dfrac{60}{2\pi}\,\dfrac{T_{\mathrm{disp}}}{J_{\mathrm{eq}}}\;[\mathrm{rpm/s}]")
st.caption(r"Tasa de cambio de rpm producida por el par disponible.")

st.latex(r"J_{\mathrm{eq}} = J_m + J_{\mathrm{driver}} + \dfrac{J_{\mathrm{driven}}+J_{\mathrm{imp}}}{r^2}")
st.caption(r"Inercia equivalente vista en el eje del motor. r = n_{\mathrm{motor}}/n_{\mathrm{bomba}}.")

st.latex(r"t_{\mathrm{par}}=\dfrac{\Delta n}{\dot{n}_{\mathrm{torque}}}, \qquad"
         r"t_{\mathrm{rampa}}=\dfrac{\Delta n}{\mathrm{rampa}_{\mathrm{VDF}}}, \qquad"
         r"t_{\mathrm{final}}=\max\!\left(t_{\mathrm{par}},t_{\mathrm{rampa}}\right)")
st.caption("Tiempos por límite de par y por límite de rampa del VDF; el tiempo real es el mayor de ambos.")

# -----------------------------------
# Entradas compactas
# -----------------------------------
ci, cf, ct = st.columns(3)
with ci:
    n_ini_m = st.number_input("Velocidad Motor inicial [rpm]", value=float(n_min), step=1.0, format="%.2f")
with cf:
    n_fin_m = st.number_input("Velocidad Motor final [rpm]", value=float(n_max), step=1.0, format="%.2f")
with ct:
    T_disp = st.number_input("Par disponible [Nm]", value=float(T_nom), step=1.0, format="%.2f")

rampa_vdf = st.sidebar.number_input("Rampa VDF [rpm/s] (motor)", min_value=1.0, value=300.0, step=1.0, format="%.2f")

# -----------------------------------
# Paso 1) Inercia equivalente (con sustitución)
# -----------------------------------
c1, c2 = st.columns(2)
with c1:
    st.markdown("**1) Inercia equivalente en el eje del motor**")
    st.latex(r"J_{\mathrm{eq}} = J_m + J_{\mathrm{driver}} + \dfrac{J_{\mathrm{driven}}+J_{\mathrm{imp}}}{r^2}")
with c2:
    st.markdown("**Sustitución:**")
    st.latex(
        rf"J_{{\mathrm{{eq}}}} = {J_m:.2f} + {J_drv:.2f} + \dfrac{{{J_drn:.2f}+{J_imp:.2f}}}{{({r:.2f})^2}}"
        rf" \;\Rightarrow\; {J_eq:.2f}\;\mathrm{{kg\cdot m^2}}"
    )

# -----------------------------------
# Paso 2) Aceleración debida al par (sin hidráulica)
# -----------------------------------
dn = max(n_fin_m - n_ini_m, 0.0)
n_dot = (60.0/(2.0*math.pi))*T_disp/max(J_eq,1e-12)   # rpm/s
t_par = dn/max(n_dot,1e-12)
t_ramp = dn/max(rampa_vdf,1e-12)
t_final = max(t_par, t_ramp)

c3, c4 = st.columns(2)
with c3:
    st.markdown("**2) Aceleración por par**")
    st.latex(r"\dot{n}_{\mathrm{torque}}=\dfrac{60}{2\pi}\,\dfrac{T_{\mathrm{disp}}}{J_{\mathrm{eq}}}")
with c4:
    st.markdown("**Sustitución:**")
    st.latex(
        rf"\dot{{n}}_{{\mathrm{{torque}}}}=\frac{{60}}{{2\pi}}\cdot\frac{{{T_disp:.2f}}}{{{J_eq:.2f}}}"
        rf"= {n_dot:.2f}\;\mathrm{{rpm/s}}"
    )

# -----------------------------------
# Paso 3) Tiempos (par / rampa / final)
# -----------------------------------
c5, c6 = st.columns(2)
with c5:
    st.markdown("**3) Tiempos de respuesta**")
    st.latex(r"t_{\mathrm{par}}=\dfrac{\Delta n}{\dot{n}_{\mathrm{torque}}},\qquad"
             r"t_{\mathrm{rampa}}=\dfrac{\Delta n}{\mathrm{rampa}_{\mathrm{VDF}}},\qquad"
             r"t_{\mathrm{final}}=\max\!\left(t_{\mathrm{par}},t_{\mathrm{rampa}}\right)")
with c6:
    st.markdown("**Sustitución:**")
    st.latex(rf"\Delta n = {dn:.2f}\;\mathrm{{rpm}}")
    st.latex(rf"t_{{\mathrm{{par}}}}=\frac{{{dn:.2f}}}{{{n_dot:.2f}}}= {t_par:.2f}\;\mathrm{{s}}")
    st.latex(rf"t_{{\mathrm{{rampa}}}}=\frac{{{dn:.2f}}}{{{rampa_vdf:.2f}}}= {t_ramp:.2f}\;\mathrm{{s}}")

st.markdown("---")

# -----------------------------------
# Resumen alineado (misma línea, símbolos idénticos y en verde)
# -----------------------------------
r1, r2, r3, r4, r5 = st.columns(5)
with r1:
    st.latex(rf"\color{{green}}{{\Delta n = {dn:.2f}\ \mathrm{{rpm}}}}")
with r2:
    st.latex(rf"\color{{green}}{{\dot{{n}}_{{\mathrm{{torque}}}}= {n_dot:.2f}\ \mathrm{{rpm/s}}}}")
with r3:
    st.latex(rf"\color{{green}}{{t_{{\mathrm{{par}}}}= {t_par:.2f}\ \mathrm{{s}}}}")
with r4:
    st.latex(rf"\color{{green}}{{t_{{\mathrm{{rampa}}}}= {t_ramp:.2f}\ \mathrm{{s}}}}")
with r5:
    st.latex(rf"\boxed{{\color{{green}}{{t_{{\mathrm{{final}}}}= {t_final:.2f}\ \mathrm{{s}}}}}}")

st.caption("En esta sección se considera solo la respuesta inercial del tren motriz (sin par hidráulico).")

# -----------------------------------
# (Opcional) Enlace al modelo hidráulico
# -----------------------------------
with st.expander("Extensión opcional: incluir la curva del sistema (H(Q)) y la eficiencia η(Q)"):
    st.markdown(
        "- Define la **curva del sistema**: " +
        r"$H(Q)=H_0+K\,Q^2$." + "\n" +
        "- Define la **eficiencia** vs. caudal: " +
        r"$\eta(Q)=\eta_a+\eta_b Q+\eta_c Q^2$." + "\n" +
        "- Con potencia hidráulica " +
        r"$P_h=\rho g\,Q\,H/\eta$ y $P=\tau\,\omega$, resulta " +
        r"$\tau_{\mathrm{load}}(Q,\omega)$.\n" +
        "- Integra la ecuación de movimiento " +
        r"$J_{\mathrm{eq}}\dot{\omega}=T_{\mathrm{disp}}-\tau_{\mathrm{load}}(\omega)$ " +
        "para obtener el tiempo real con hidráulica."
    )


# ------------------------------
# Modelos auxiliares para hidráulica
# ------------------------------
g = 9.81

def eta_of_Qn(Q_m3h: float, n_rpm: float, vals: Dict[str,float]) -> float:
    # adimensional y con escalado en rpm opcional
    if vals["Qref"] <= 0:
        return max(min(0.72, vals["eta_max_clip"]), vals["eta_min_clip"])
    qhat = Q_m3h / vals["Qref"]
    eta = vals["eta_a"] + vals["eta_b"]*qhat + vals["eta_c"]*(qhat**2)
    if vals["eta_beta"] != 0 and vals["n_nom"]>0:
        eta *= (n_rpm/vals["n_nom"])**(vals["eta_beta"])
    return float(min(max(eta, vals["eta_min_clip"]), vals["eta_max_clip"]))

def H_of_Q(Q_m3h: float, vals: Dict[str,float]) -> float:
    Qs = Q_m3h/3600.0
    return vals["H0"] + vals["K"]*(Qs**2)

def Q_of_n_b(n_b: float, vals: Dict[str,float]) -> float:
    # Afinidad Q ∝ n (anclado en n_ref = n_nom con Q_ref)
    if vals["n_nom"] <= 0:
        return 0.0
    Q = vals["Qref"] * (n_b/vals["n_nom"])
    # clamp al rango nominal de operación
    if vals["Qmax"] > vals["Qmin"] > 0:
        Q = min(max(Q, vals["Qmin"]), vals["Qmax"])
    return max(Q, 0.0)

def T_load_eq_from_nm(n_m: float, vals: Dict[str,float]) -> Tuple[float, float, float, float]:
    """Devuelve (T_load_eq [Nm en motor], Q [m3/h], H [m], eta_used)."""
    r = vals["r_tr"]
    n_b = n_m / r
    Q = Q_of_n_b(n_b, vals)
    H = H_of_Q(Q, vals)
    eta = eta_of_Qn(Q, n_b, vals)
    Qs = Q/3600.0
    P_h = vals["rho"] * g * Qs * H  # W
    P_shaft = P_h / max(eta, 1e-9)  # W
    omega_p = 2.0*math.pi*n_b/60.0
    T_pump = P_shaft / max(omega_p, 1e-9)  # Nm (eje bomba)
    T_eq = T_pump / r                      # reflejado al motor
    return T_eq, Q, H, eta

def integrate_with_hyd(J_eq: float, T_disp: float, n_m_i: float, n_m_f: float, vals: Dict[str,float], step_rpm: float = 5.0) -> Dict[str, float]:
    """Integración por escalones en n_m (de i → f).
       Si T_disp <= T_load_eq, se detiene (bloqueado)."""
    reverse = False
    if n_m_f < n_m_i:
        n_m_i, n_m_f = n_m_f, n_m_i
        reverse = True

    n_list = np.arange(n_m_i, n_m_f+step_rpm, step_rpm, dtype=float)
    t_total = 0.0
    q_ini = None
    q_fin = None
    blocked_at = None

    for k in range(len(n_list)-1):
        n1 = float(n_list[k])
        n2 = float(n_list[k+1])
        dnm = n2 - n1
        T_load, Q1, H1, eta1 = T_load_eq_from_nm(n1, vals)
        if q_ini is None:
            q_ini = Q1
        q_fin = Q1
        T_net = T_disp - T_load
        if T_net <= 0:
            blocked_at = n1
            break
        n_dot = (60.0/(2.0*math.pi))*(T_net/max(J_eq,1e-9))  # rpm/s
        dt = dnm/max(n_dot,1e-9)
        t_total += dt

    if reverse:
        # si el usuario eligió rango descendente, el tiempo es el mismo (simetría del cálculo) y Q_ini/Q_fin se invierten
        q_ini, q_fin = q_fin, q_ini

    return dict(
        t_total=t_total,
        Q_ini=q_ini if q_ini is not None else 0.0,
        Q_fin=q_fin if q_fin is not None else 0.0,
        blocked_at=blocked_at
    )

# ------------------------------
# Sección 4 – Dinámica con hidráulica (rango de rpm de bomba)
# ------------------------------
st.subheader("4) Tiempo de reacción con hidráulica (rango de velocidad de la bomba)")

# sliders compactos en columnas
c41, c42 = st.columns((1,1))
with c41:
    n_b_min = vals["n_min"]/vals["r_tr"]
    n_b_max = vals["n_max"]/vals["r_tr"]
    rng = st.slider(
        "Rango de velocidad de bomba [rpm]",
        min_value=float(n_b_min), max_value=float(n_b_max),
        value=(float(n_b_min), float(n_b_max))
    )
with c42:
    step_rpm = st.number_input("Paso de integración [rpm (motor)]", min_value=1.0, value=5.0, step=1.0)

n_b_ini, n_b_fin = rng
n_m_ini = n_b_ini * vals["r_tr"]
n_m_fin = n_b_fin * vals["r_tr"]

st.markdown(f"- **Velocidad motor equivalente [rpm]**: {fmt2(n_m_ini,'rpm')} → {fmt2(n_m_fin,'rpm')}")

st.latex(r"""
P_h=\rho g Q_s H(Q),\quad Q_s=\frac{Q}{3600},\qquad
T_{\mathrm{pump}}=\frac{P_h}{\omega_p},\quad \omega_p=\frac{2\pi n_{\mathrm{bomba}}}{60},\qquad
T_{\mathrm{load,eq}}=\frac{T_{\mathrm{pump}}}{r}.
""")

hyd = integrate_with_hyd(J_eq, T_disp, n_m_ini, n_m_fin, vals, step_rpm=step_rpm)

cH1, cH2, cH3 = st.columns((1,1,1))
with cH1: st.metric("Q_ini [m³/h]", fmt2(hyd["Q_ini"]))
with cH2: st.metric("Q_fin [m³/h]", fmt2(hyd["Q_fin"]))
with cH3: st.metric("ΔQ [m³/h]", fmt2(hyd["Q_fin"]-hyd["Q_ini"]))

if hyd["blocked_at"] is not None:
    st.error(f"No hay margen de par en **{fmt2(hyd['blocked_at'],'rpm')} (motor)**. El rango no se completa.")
else:
    st.success(f"**Tiempo con hidráulica** para el rango seleccionado: **{fmt2(hyd['t_total'],'s')}**")

st.markdown("---")

# ------------------------------
# Sección 5 – Exportación / Reportes
# ------------------------------
st.subheader("5) Exportación")

def times_no_hyd_tag(vals_i: Dict[str,float], rampa_vdf: float) -> Dict[str,float]:
    J_eq_i = vals_i["J_m"] + vals_i["J_drv"] + (vals_i["J_drn"]+vals_i["J_imp"])/(vals_i["r_tr"]**2)
    dn_i, n_dot_i, t_par_i, t_ramp_i, t_fin_i = times_no_hyd(
        J_eq_i, vals_i["T_nom"], vals_i["n_min"], vals_i["n_max"], rampa_vdf
    )
    return dict(J_eq=J_eq_i, dn=dn_i, n_dot=n_dot_i, t_par=t_par_i, t_rampa=t_ramp_i, t_final=t_fin_i)

def build_summary_table(df: pd.DataFrame, rampa_vdf: float) -> pd.DataFrame:
    rows = []
    for _, rw in df.iterrows():
        # leer por fila
        vals_i = read_inputs(rw)
        base = times_no_hyd_tag(vals_i, rampa_vdf)
        rows.append({
            "TAG": str(rw[COL_TAG]),
            "r": vals_i["r_tr"],
            "J_m": vals_i["J_m"],
            "J_driver": vals_i["J_drv"],
            "J_driven": vals_i["J_drn"],
            "J_imp": vals_i["J_imp"],
            "J_eq": base["J_eq"],
            "n_motor_min": vals_i["n_min"],
            "n_motor_max": vals_i["n_max"],
            "Δn": base["dn"],
            "n_dot_torque": base["n_dot"],
            "t_par": base["t_par"],
            "t_rampa": base["t_rampa"],
            "t_final_sin": base["t_final"]
        })
    out = pd.DataFrame(rows)
    num_cols = [c for c in out.columns if c != "TAG"]
    out[num_cols] = out[num_cols].astype(float).round(2)
    return out

if st.button("Generar resumen por TAG (CSV)"):
    tbl = build_summary_table(df, rampa_vdf)
    st.dataframe(tbl, use_container_width=True)
    csv = tbl.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Descargar CSV", data=csv, file_name="resumen_por_TAG.csv", mime="text/csv")

# Reporte del TAG actual (parámetros incluidos y calculados)
rep_dict = {
    "TAG": tag,
    "r": vals["r_tr"],
    "J_m": round(vals["J_m"],2),
    "J_driver": round(vals["J_drv"],2),
    "J_driven": round(vals["J_drn"],2),
    "J_imp": round(vals["J_imp"],2),
    "J_eq": round(J_eq,2),
    "n_motor_min": round(vals["n_min"],2),
    "n_motor_max": round(vals["n_max"],2),
    "n_bomba_min": round(vals["n_min"]/vals["r_tr"],2),
    "n_bomba_max": round(vals["n_max"]/vals["r_tr"],2),
    "T_nom": round(vals["T_nom"],2),
    "Rampa VDF [rpm/s]": round(rampa_vdf,2),
    "t_final_sin [s]": round(t_fin_sin,2),
    "Rango bomba [rpm]": f"{n_b_ini:.2f} → {n_b_fin:.2f}",
    "Q_ini [m3/h]": round(hyd["Q_ini"],2),
    "Q_fin [m3/h]": round(hyd["Q_fin"],2),
    "ΔQ [m3/h]": round(hyd["Q_fin"]-hyd["Q_ini"],2),
    "t_con_hidraulica [s]": round(hyd["t_total"],2),
    "bloqueado_en [rpm motor]": (None if hyd["blocked_at"] is None else round(hyd["blocked_at"],2)),
    "H0 [m]": round(vals["H0"],2),
    "K [m s^2]": round(vals["K"],2),
    "rho [kg/m3]": round(vals["rho"],2),
    "eta_a": round(vals["eta_a"],4),
    "eta_b": round(vals["eta_b"],4),
    "eta_c": round(vals["eta_c"],4),
    "eta_beta": round(vals["eta_beta"],2),
    "eta_min_clip": round(vals["eta_min_clip"],2),
    "eta_max_clip": round(vals["eta_max_clip"],2)
}
rep = pd.DataFrame([rep_dict])

st.dataframe(rep, use_container_width=True)
rep_csv = rep.to_csv(index=False).encode("utf-8-sig")
st.download_button("Descargar reporte del TAG actual (CSV)", data=rep_csv, file_name=f"reporte_{tag}.csv", mime="text/csv")
