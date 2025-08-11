
import streamlit as st
import pandas as pd, numpy as np, math, re, os

st.set_page_config(page_title="Tiempos de reacción – Bombas con VDF (v4.1)", layout="wide")

# ---------------- Sidebar ----------------
st.sidebar.header("Parámetros globales")
ramp_motor = st.sidebar.number_input("Rampa VDF (rpm/s en motor)", min_value=10.0, max_value=5000.0, value=300.0, step=10.0)

st.title("Tiempos de reacción – Modelo individual por TAG (v4.1)")

# ---------------- Helpers ----------------
G = 9.80665

def _to_float_mixed(x):
    """Convierte strings con formato '1.234,56' o '1234,56' a float. Deja pasar floats/ints."""
    if x is None or (isinstance(x, float) and np.isnan(x)): return np.nan
    if isinstance(x, (int, float)): return float(x)
    s = str(x).strip()
    if s == "": return np.nan
    if ("," in s) and ("." in s):
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except:
        return np.nan

def ring_J(m_kg, Ro_m, Ri_m=0.0):
    return m_kg*(Ro_m**2 + Ri_m**2)/2.0

def inertia_disc_ring(mass_kg, D_mm):
    R = (D_mm/1000.0)/2.0
    return 0.5*mass_kg*(R**2), 1.0*mass_kg*(R**2)

def sheave_inertia_est(series, od_in, grooves, weight_lb=None):
    """Si hay peso (lb), úsalo; si no, estimación geométrica sencilla."""
    inch_to_m = 0.0254; lb_to_kg = 0.45359237
    if weight_lb and not np.isnan(weight_lb) and weight_lb > 0:
        m = weight_lb*lb_to_kg
        Ro = (od_in*inch_to_m)/2.0
        return ring_J(m, Ro), m
    # fallback
    F_in = 3.0 + 0.5*max(0,grooves-2); t_in = 0.6; rho = 7200
    Ro = (od_in*inch_to_m)/2.0; Ri = max(Ro - t_in*inch_to_m, 0.0)
    vol = math.pi*(Ro**2 - Ri**2)*(F_in*inch_to_m)
    m = rho*vol
    return ring_J(m, Ro), m

def _extract_numbers(line, unit_pattern):
    def _to_float(s):
        s = s.strip()
        if (',' in s) and ('.' in s):
            s = s.replace('.', '').replace(',', '.')
        else:
            s = s.replace(',', '.')
        return float(s)
    nums = []
    for m in re.finditer(r'([\d]+[\d\.,]*)\s*' + unit_pattern, line):
        try:
            nums.append(_to_float(m.group(1)))
        except:
            pass
    return nums

def parse_summary_points(pdf_path):
    import pdfplumber
    txt = ""
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            t = p.extract_text() or ""
            txt += t + "\n"
    m = re.search(r'(4210|4220|4230)-PU-\d{3}', txt)
    tag = m.group(0) if m else None

    m_rho = re.search(r'Densidad\s+de\s+la\s+pulpa\s+([\d\.,]+)\s*kg/m', txt)
    SG = None
    if m_rho:
        rho = float(m_rho.group(1).replace('.', '').replace(',', '.'))
        SG = rho/1000.0

    m_mu = re.search(r'Viscosidad\s+de\s+la\s+pulpa\s+([\d\.,]+)\s*Pa', txt)
    mu = float(m_mu.group(1).replace(',', '.')) if m_mu else None

    m_ff = re.search(r'Factor\s+de\s+espuma\s+([\d\.,]+)', txt)
    FF = float(m_ff.group(1).replace(',', '.')) if m_ff else None

    m_tdh = re.search(r'TDH\s+([^\n]+)', txt)
    m_q   = re.search(r'Caudal\s+requerido\s+([^\n]+)', txt)
    m_rpm = re.search(r'Velocidad\s+operacional\s+([^\n]+)', txt)
    m_eta = re.search(r'Eficiencia\s+operacional\s+([^\n]+)', txt)
    m_pkW = re.search(r'Potencia\s+absorbida\s+([^\n]+)', txt)

    Qs = _extract_numbers(m_q.group(1), r'm³/h') if m_q else []
    Hs = _extract_numbers(m_tdh.group(1), r'm') if m_tdh else []
    rpms = _extract_numbers(m_rpm.group(1), r'rpm') if m_rpm else []
    etas = _extract_numbers(m_eta.group(1), r'%') if m_eta else []
    Ps   = _extract_numbers(m_pkW.group(1), r'kW') if m_pkW else []

    labels = ["Mínimo","Vel. Máx. Actual","Nominal Base","Optimizado P90","Optimizado 01/5-2-1","Potencia Máxima"]
    n = max(len(Qs), len(Hs), len(rpms), len(etas), len(Ps))
    points = []
    for i in range(n):
        points.append({
            "label": labels[i] if i < len(labels) else f"P{i+1}",
            "Q_m3h": Qs[i] if i < len(Qs) else None,
            "H_m":  Hs[i] if i < len(Hs) else None,
            "rpm":  rpms[i] if i < len(rpms) else None,
            "eta":  (etas[i]/100.0) if i < len(etas) else None,
            "P_kW": Ps[i] if i < len(Ps) else None
        })

    q = [p["Q_m3h"]/3600.0 for p in points if p["Q_m3h"] and p["H_m"]]
    h = [p["H_m"] for p in points if p["Q_m3h"] and p["H_m"]]
    H0, K = None, None
    if len(q) >= 2:
        X = np.array([np.ones(len(q)), np.square(q)]).T
        beta, *_ = np.linalg.lstsq(X, np.array(h), rcond=None)
        H0, K = float(beta[0]), float(beta[1])

    return {"tag": tag, "SG": SG, "mu": mu, "FF": FF, "points": points, "H0": H0, "K": K}

def tload_from_system(n_rpm, Qref_m3h, H0, K, nref_rpm, SG, eta):
    rho = 1000.0*SG
    alpha = (Qref_m3h/3600.0)/max(nref_rpm,1e-6)
    n = np.array(n_rpm, dtype=float)
    Qs = alpha * n
    H = H0 + K*(Qs**2)
    Ph = rho*G*Qs*H/ max(eta,1e-6)
    omega = (2.0*math.pi/60.0)*n
    return Ph/np.maximum(omega,1e-6)

def integrate_time(n1, n2, ramp_motor_rpm_s, J_eq, T_avail_fun, T_load_fun, steps=800):
    t=0.0; dn=(n2-n1)/steps
    for i in range(steps):
        nmid = n1+(i+0.5)*dn
        Tm = max(T_avail_fun(nmid) - T_load_fun(nmid), 0.0)
        alpha = Tm/max(J_eq,1e-9)
        accel_rpm_s_torque = (60.0/(2.0*math.pi))*alpha
        accel_rpm_s = min(accel_rpm_s_torque, ramp_motor_rpm_s)
        if accel_rpm_s <= 0: return float('inf')
        t += dn/accel_rpm_s
    return t

# ---------------- Inputs ----------------
st.markdown("## 1) Archivos de entrada")

up_trans = st.file_uploader("CSV de transmisiones (sheaves) con pesos", type=["csv"])
if up_trans:
    sheaves_df = pd.read_csv(up_trans)
else:
    try:
        sheaves_df = pd.read_csv("sheaves_default.csv")
    except Exception:
        sheaves_df = pd.DataFrame(columns=["TAG","driver_series","driver_od_in","driver_grooves","driver_weight_lb","driven_series","driven_od_in","driven_grooves","driven_weight_lb"])

# Coerción de numéricos (OD, grooves, weights)
for c in ["driver_od_in","driven_od_in","driver_grooves","driven_grooves","driver_weight_lb","driven_weight_lb"]:
    if c in sheaves_df.columns:
        sheaves_df[c] = sheaves_df[c].apply(_to_float_mixed)

st.dataframe(sheaves_df, use_container_width=True)

st.markdown("**Inputs mecánicos por TAG**")
mech_cols = ["TAG","MotorPower_kW","Poles","Jm_kgm2","Impeller_D_mm","Impeller_mass_kg","n_motor_min","n_motor_max","n_pump_min","n_pump_max"]
mech_default = pd.DataFrame([
    ("4210-PU-003", 37.0, 4, 0.5177, 600.0, 228.7, 738, 1475, 234, 469),
    ("4220-PU-010",200.0, 6,11.0000,1000.0, 816.2, 495,  990, 200, 400),
    ("4230-PU-011",200.0, 4, 4.4300, 750.0, 268.6, 745, 1490, 294, 588),
    ("4230-PU-015", 75.0, 4, 1.6400, 750.0, 268.6, 743, 1485, 216, 432),
    ("4230-PU-022", 90.0, 4, 2.5700, 750.0, 128.1, 745, 1489, 216, 433),
    ("4230-PU-031",110.0, 4, 2.5700, 600.0, 228.7, 745, 1489, 318, 635),
], columns=mech_cols)
up_mech = st.file_uploader("CSV de inputs mecánicos (opcional)", type=["csv"], key="mech")
if up_mech: mech_df = pd.read_csv(up_mech)
else: mech_df = mech_default.copy()
st.dataframe(mech_df, use_container_width=True)

st.markdown("**PDFs Summary** en carpeta de la app:")
pdf_list = [p for p in os.listdir() if p.startswith("Summary_") and p.endswith(".pdf")]
st.write(f"{len(pdf_list)} detectados.")

all_tags = sorted(set(sheaves_df["TAG"].dropna().astype(str)).union(set(mech_df["TAG"].dropna().astype(str))))

st.markdown("---")
st.markdown("## 2) Análisis individual por TAG")
tag_sel = st.selectbox("Selecciona el TAG", options=all_tags)
if not tag_sel: st.stop()

row_m = mech_df[mech_df["TAG"]==tag_sel].iloc[0] if (mech_df["TAG"]==tag_sel).any() else None
row_s = sheaves_df[sheaves_df["TAG"]==tag_sel].iloc[0] if (sheaves_df["TAG"]==tag_sel).any() else None

# Buscar PDF correspondiente
pdf_path = None
for p in pdf_list:
    if tag_sel in p:
        pdf_path = p; break

col1, col2 = st.columns([1,1])
with col1:
    st.subheader("Entradas del TAG")
    st.markdown("**Transmisión (con pesos si están en el CSV)**")
    st.write(row_s.to_frame().T if row_s is not None else "—")

    st.markdown("**Mecánica**")
    st.write(row_m.to_frame().T if row_m is not None else "—")

    st.markdown("**Datos de pulpa + 5 puntos (desde PDF)**")
    if pdf_path:
        pdata = parse_summary_points(pdf_path)
        pts_df = pd.DataFrame(pdata["points"])
        st.write(f"PDF: `{pdf_path}`")
        st.dataframe(pts_df, use_container_width=True, height=220)
        SG = pdata["SG"] if pdata["SG"] else 1.0; mu = pdata["mu"]; FF = pdata["FF"]
        H0 = pdata["H0"]; K = pdata["K"]
    else:
        pdata = {"points":[]}; SG, mu, FF, H0, K = 1.0, None, None, None, None
        st.info("No se encontró PDF para este TAG.")

with col2:
    st.subheader("Curva de sistema ajustada")
    if H0 is not None and K is not None:
        st.metric("H0 (m)", f"{H0:.2f}"); st.metric("K (m/(m³/s)²)", f"{K:.0f}")
    else:
        st.info("No fue posible ajustar H0/K automáticamente. Edita valores:")
        H0 = st.number_input("H0 (m)", 0.0, 50.0, 5.0, 0.1)
        K  = st.number_input("K (m/(m³/s)²)", 0.0, 50000.0, 15000.0, 100.0)

    # Duty de referencia (Nominal Base si existe)
    Qref, Href, eta_ref, nref = None, None, None, None
    for p in pdata["points"]:
        if "Nominal" in p["label"] or "Base" in p["label"]:
            Qref, Href, eta_ref, nref = p["Q_m3h"], p["H_m"], p["eta"], p["rpm"]
            break
    if Qref is None:
        for p in pdata["points"]:
            if p["Q_m3h"] and p["H_m"]:
                Qref, Href, eta_ref, nref = p["Q_m3h"], p["H_m"], (p["eta"] or 0.7), (p["rpm"] or row_m["n_motor_max"])
                break
    if Qref is None: Qref, Href, eta_ref, nref = 500.0, 10.0, 0.7, float(row_m["n_motor_max"])

    st.write(pd.DataFrame([{"Q_ref_m3h":Qref,"H_ref_m":Href,"eta_ref":eta_ref,"nref_rpm":nref,"SG":SG,"FF":FF,"mu":mu}]))

# ---------------- Inercias ----------------
st.markdown("## 3) Inercias y transmisión")
if row_s is not None and row_m is not None:
    r_trans = float(row_s["driven_od_in"])/float(row_s["driver_od_in"])
    st.write(f"**r = driven/driver = {r_trans:.3f}**")

    Jm = float(row_m["Jm_kgm2"])
    J_imp_disc, J_imp_ring = inertia_disc_ring(float(row_m["Impeller_mass_kg"]), float(row_m["Impeller_D_mm"]))
    inertia_model = st.radio("Inercia del impulsor", ["disco","aro"], horizontal=True, index=0)
    J_imp = J_imp_disc if inertia_model=="disco" else J_imp_ring

    # Pesos desde CSV (si existen)
    w_driver_lb_csv = row_s["driver_weight_lb"] if "driver_weight_lb" in row_s else np.nan
    w_driven_lb_csv = row_s["driven_weight_lb"] if "driven_weight_lb" in row_s else np.nan

    with st.expander("Poleas (peso – usar CSV o sobrescribir)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            w_driver_lb = st.number_input("Peso polea motriz (lb)", 0.0, 2000.0, float(_to_float_mixed(w_driver_lb_csv) if not pd.isna(w_driver_lb_csv) else 0.0), 1.0)
        with c2:
            w_driven_lb = st.number_input("Peso polea conducida (lb)", 0.0, 3000.0, float(_to_float_mixed(w_driven_lb_csv) if not pd.isna(w_driven_lb_csv) else 0.0), 1.0)
    J_driver, m_driver = sheave_inertia_est(row_s["driver_series"], float(row_s["driver_od_in"]), int(row_s["driver_grooves"]), weight_lb=w_driver_lb)
    J_driven, m_driven = sheave_inertia_est(row_s["driven_series"], float(row_s["driven_od_in"]), int(row_s["driven_grooves"]), weight_lb=w_driven_lb)

    belt_mass = st.number_input("Masa equivalente de correas (kg/polea)", 0.0, 50.0, 5.0, 0.5)
    J_driver += ring_J(belt_mass, (float(row_s["driver_od_in"])*0.0254)/2.0)
    J_driven += ring_J(belt_mass, (float(row_s["driven_od_in"])*0.0254)/2.0)

    # Fluido
    R_m = (float(row_m["Impeller_D_mm"])/1000.0)/2.0
    rho = 1000.0*(SG if SG else 1.0)
    m_fluid = rho*(math.pi*R_m**2 * 0.05)
    k_fluid = st.number_input("k_fluid (J_fluid = k · m_fluid · R^2)", 0.0, 1.0, 0.0, 0.05)
    J_fluid = k_fluid * m_fluid * R_m**2

    J_eq = Jm + J_driver + (r_trans**2)*(J_imp + J_driven + J_fluid)

    st.write(pd.DataFrame([{
        "Jm":Jm,"J_driver":J_driver,"J_driven":J_driven,"J_imp":J_imp,"J_fluid":J_fluid,"J_eq (motor-side)":J_eq,
        "w_driver_lb_used": w_driver_lb, "w_driven_lb_used": w_driven_lb
    }]))
else:
    st.warning("Faltan datos para inercias.")

# ---------------- Par disponible y carga ----------------
st.markdown("## 4) Par disponible (estimado) y par de carga")
if row_m is not None:
    T_nom = 9550.0*float(row_m["MotorPower_kW"])/max(float(row_m["n_motor_max"]), 1.0)
    st.write(pd.DataFrame([{"T_nom_est [Nm]":T_nom,"n_nom [rpm]":float(row_m["n_motor_max"])}]))
    T_avail_fun = lambda n: T_nom
else:
    T_avail_fun = lambda n: 0.0

if H0 is not None and K is not None:
    T_load_fun = lambda n: tload_from_system(n, Qref, H0, K, (nref or float(row_m["n_motor_max"])), (SG or 1.0), (eta_ref or 0.7))
else:
    T_load_fun = lambda n: 0.0

# ---------------- Tiempos ----------------
st.markdown("## 5) Tiempo de reacción")
if row_m is not None and row_s is not None:
    n1 = float(row_m["n_motor_min"]); n2 = float(row_m["n_motor_max"])
    t_par = integrate_time(n1, n2, ramp_motor, J_eq, T_avail_fun, T_load_fun)
    pump_accel = ramp_motor / (float(row_s["driven_od_in"])/float(row_s["driver_od_in"]))
    t_ramp_only = (float(row_m["n_pump_max"]) - float(row_m["n_pump_min"])) / pump_accel if pump_accel>0 else float("inf")
    t_final = max(t_par, t_ramp_only)
    st.write(pd.DataFrame([{"t_ramp_only [s]":t_ramp_only,"t_par [s]":t_par,"t_final [s]":t_final}]))

st.markdown("---")
with st.expander("Explicación rápida de cada valor y fórmula"):
    st.markdown(r"""
- **H0, K**: ajuste con 5 puntos Q–H del PDF → $$H(Q)=H_0+KQ^2$$.\
- **Q≈αn**: con \(Q_\mathrm{ref}, n_\mathrm{ref}\) del punto nominal del PDF.\
- **T_load**: $$P=\rho g Q H/\eta,\ \ T=P/\omega$$ con \(\rho=1000\cdot SG\).\
- **J_eq**: $$J_m+J_{driver}+r^2(J_{imp}+J_{driven}+J_{fluido})$$.\
- **T_avail** (estimado): $$9550\,P(kW)/n(rpm)$$.\
- **t_par**: integra \(\alpha(n)=\min\{\alpha_{torque},\dot n_{motor}\}\) desde \(n_i\) a \(n_f\).\
- **t_rampa**: piso por rampa del variador en la bomba.\
- **t_final** = máx(t_par, t_rampa).
""")
