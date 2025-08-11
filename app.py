import streamlit as st
import pandas as pd
import numpy as np
import math

st.set_page_config(page_title="Tiempo de reacción bombas con VDF – v3", layout="wide")

G = 9.80665

def inertia_disc_ring(mass_kg, D_mm):
    R = (D_mm/1000.0)/2.0
    J_disc = 0.5 * mass_kg * (R**2)
    J_ring = 1.0 * mass_kg * (R**2)
    return J_disc, J_ring

def tload_system_from_params(n_rpm, Qref_m3h, H0, K, nref_rpm, SG, eta):
    rho = 1000.0*SG
    alpha = (Qref_m3h/3600.0)/max(nref_rpm,1e-6)
    n = np.array(n_rpm, dtype=float)
    Qs = alpha * n
    H = H0 + K * (Qs**2)
    Ph = rho*G*Qs*H / max(eta,1e-6)
    omega = (2.0*math.pi/60.0)*n
    return Ph / np.maximum(omega,1e-6)

def compute_Jeq_motor_side(r, Jm, J_imp, J_driver, J_driven, J_fluid=0.0):
    return Jm + J_driver + (r**2)*(J_imp + J_driven + J_fluid)

def ring_J(m_kg, Ro_m, Ri_m=0.0):
    return m_kg*(Ro_m**2 + Ri_m**2)/2.0

def sheave_inertia(series, od_in, grooves, bushing_series=None, shaft_mm=None, weight_lb=None):
    lb_to_kg = 0.45359237; inch_to_m = 0.0254
    if weight_lb is not None:
        m_kg = weight_lb*lb_to_kg
    else:
        F_in = 3.0 + 0.5*max(0,grooves-2); t_in = 0.6; rho = 7200
        Ro = (od_in*inch_to_m)/2.0; Ri = max(Ro - t_in*inch_to_m, 0.0)
        vol = math.pi*(Ro**2 - Ri**2)*(F_in*inch_to_m)
        m_kg = rho*vol
    Ro = (od_in*inch_to_m)/2.0
    return ring_J(m_kg, Ro), m_kg

def integrate_time_motor(n1, n2, ramp_motor_rpm_s, J_eq, T_avail_fun, T_load_fun, steps=600):
    t = 0.0; dn = (n2-n1)/steps
    for i in range(steps):
        n_mid = n1 + (i+0.5)*dn
        Tm = max(T_avail_fun(n_mid) - T_load_fun(n_mid), 0.0)
        alpha = Tm / max(J_eq,1e-9)
        accel_rpm_s_torque = (60.0/(2.0*math.pi))*alpha
        accel_rpm_s = min(accel_rpm_s_torque, ramp_motor_rpm_s)
        if accel_rpm_s <= 0: return float('inf')
        t += dn/accel_rpm_s
    return t

st.sidebar.header("Parámetros globales")
ramp_motor = st.sidebar.number_input("Rampa VDF (rpm/s en motor)", 10.0, 5000.0, 300.0, 10.0)
overload_pu = st.sidebar.number_input("Sobrecarga de par (pu)", 0.5, 3.0, 1.0, 0.1)
torque_nominal_by_tag = st.sidebar.text_area("Par nominal por TAG (Nm) – JSON", value='{"4210-PU-003":240, "4220-PU-010":1930, "4230-PU-011":1282, "4230-PU-015":482, "4230-PU-022":577, "4230-PU-023":577, "4230-PU-024":577, "4230-PU-031":706}')

st.title("Tiempo de reacción – modelo con densidad, sistema e inercias de transmisión (v3)")

st.subheader("1) Dataset inicial (SG, espuma, viscosidad, duty, H0, K)")
up_init = st.file_uploader("Sube initial_dataset.csv (o usa el que viene por defecto)", type=["csv"], key="init")
if up_init: base_df = pd.read_csv(up_init)
else: base_df = pd.read_csv("initial_dataset.csv")
st.dataframe(base_df, use_container_width=True)

st.subheader("2) Tabla de equipos (transmisión y geometría)")
st.caption("Incluye diámetros de poleas, serie 5V/8V, ranuras, bushing y diámetro de eje. Puedes cargar sheaves_default.csv o editar aquí.")
up_sh = st.file_uploader("Sube sheaves_default.csv (o usa el que viene por defecto)", type=["csv"], key="sheaves")
if up_sh: sheaves_df = pd.read_csv(up_sh)
else: sheaves_df = pd.read_csv("sheaves_default.csv")
sheaves_df = st.data_editor(sheaves_df, use_container_width=True, num_rows="dynamic")
st.download_button("Descargar sheaves CSV", sheaves_df.to_csv(index=False).encode("utf-8"), "sheaves_edited.csv", "text/csv")
sheaves_df["r"] = sheaves_df["driven_od_in"]/sheaves_df["driver_od_in"]
st.write("Relaciones derivadas de poleas:", sheaves_df[["TAG","driver_od_in","driven_od_in","r"]])

st.subheader("3) Cálculo de tiempos por TAG")
mech_cols = ["TAG","Jm_kgm2","Impeller_D_mm","Impeller_mass_kg","n_motor_min","n_motor_max","n_pump_min","n_pump_max"]
mech_default = pd.DataFrame([
    ("4210-PU-003", 0.5177, 600.0, 228.7, 738, 1475, 234, 469),
    ("4220-PU-010",11.0000,1000.0,816.2, 495,  990, 200, 400),
    ("4230-PU-011", 4.4300, 750.0, 268.6, 745, 1490, 294, 588),
    ("4230-PU-015", 1.6400, 750.0, 268.6, 743, 1485, 216, 432),
    ("4230-PU-022", 2.5700, 750.0, 128.1, 745, 1489, 216, 433),
    ("4230-PU-031", 2.5700, 600.0, 228.7, 745, 1489, 318, 635),
], columns=mech_cols)
mech_df = st.data_editor(mech_default, use_container_width=True, num_rows="dynamic")
st.download_button("Descargar mecánica CSV", mech_df.to_csv(index=False).encode("utf-8"), "mechanical_inputs.csv", "text/csv")

df = mech_df.merge(base_df, on="TAG", how="left").merge(sheaves_df, on="TAG", how="left")

import json as _json
try: T_nom_map = _json.loads(torque_nominal_by_tag)
except Exception as e:
    st.error(f"JSON de torques inválido: {e}"); T_nom_map = {}

inertia_model = st.selectbox("Modelo de inercia del impulsor", ["disco","aro"], index=0)
belt_mass_factor = st.number_input("Masa equivalente de correas por polea (kg)", 0.0, 50.0, 5.0, 0.5)
fluid_J_coeff = st.number_input("Coeff. de inercia del fluido k (J_fluid = k·m_fluid·R^2)", 0.0, 1.0, 0.0, 0.05)

rows=[]
for _, r in df.iterrows():
    tag = r["TAG"]; T_nom = float(T_nom_map.get(tag, np.nan))
    if np.isnan(T_nom): 
        rows.append({"TAG":tag, "status":"Falta T_nom"}); continue
    D_mm = float(r["Impeller_D_mm"]); m_imp = float(r["Impeller_mass_kg"]); R_m = (D_mm/1000.0)/2.0
    J_imp_disc, J_imp_ring = inertia_disc_ring(m_imp, D_mm); J_imp = J_imp_disc if inertia_model=="disco" else J_imp_ring
    # Sheaves inertia + belts
    def _sheave_J(series, od_in, grooves):
        J, m = sheave_inertia(series, float(od_in), int(grooves))
        J += ring_J(belt_mass_factor, (float(od_in)*0.0254)/2.0)
        return J
    J_driver = _sheave_J(r["driver_series"], r["driver_od_in"], r["driver_grooves"])
    J_driven = _sheave_J(r["driven_series"], r["driven_od_in"], r["driven_grooves"])
    r_tr = float(r["r"]); rho = 1000.0*float(r["SG"]) if not pd.isna(r["SG"]) else 1000.0
    m_fluid = rho * (math.pi*R_m**2 * 0.05)  # ancho 5 cm
    J_fluid = fluid_J_coeff * m_fluid * R_m**2
    Jm = float(r["Jm_kgm2"]); J_eq = Jm + J_driver + (r_tr**2)*(J_imp + J_driven + J_fluid)
    T_avail = overload_pu * T_nom; T_avail_fun = lambda n: T_avail
    nref = float(r["n_motor_max"]); H0 = float(r["H0_m"]) if not pd.isna(r["H0_m"]) else 0.0
    K  = float(r["K_m_per_m3s2"]) if not pd.isna(r["K_m_per_m3s2"]) else 0.0
    Qref = float(r["Q_ref_m3h"]) if not pd.isna(r["Q_ref_m3h"]) else 500.0
    eta = float(r["Eta_ref"]) if not pd.isna(r["Eta_ref"]) else 0.65
    SG  = float(r["SG"]) if not pd.isna(r["SG"]) else 1.0
    T_load_fun = lambda n: tload_system_from_params(n, Qref, H0, K, nref, SG, eta)
    n1, n2 = float(r["n_motor_min"]), float(r["n_motor_max"])
    # Integración
    def integrate(n1, n2, ramp_motor, J_eq, T_avail_fun, T_load_fun, steps=800):
        t=0.0; dn=(n2-n1)/steps
        for i in range(steps):
            nmid = n1+(i+0.5)*dn
            Tm = max(T_avail_fun(nmid)-T_load_fun(nmid), 0.0)
            alpha = Tm/max(J_eq,1e-9)
            accel = min((60.0/(2.0*math.pi))*alpha, ramp_motor)
            if accel<=0: return float("inf")
            t += dn/accel
        return t
    t_par = integrate(n1, n2, ramp_motor, J_eq, T_avail_fun, T_load_fun)
    pump_accel = ramp_motor / max(r_tr,1e-6)
    t_ramp_only = (float(r["n_pump_max"])-float(r["n_pump_min"])) / pump_accel if pump_accel>0 else float("inf")
    t_final = max(t_ramp_only, t_par)
    rows.append({"TAG":tag,"r_trans":r_tr,"J_eq [kg m2]":J_eq,"t_ramp_only [s]":t_ramp_only,"t_par [s]":t_par,"t_final [s]":t_final,
                 "SG":SG,"eta":eta,"H0":H0,"K":K,"Q_ref_m3h":Qref,"J_driver":J_driver,"J_driven":J_driven,"J_imp":J_imp,"J_fluid":J_fluid})
res = pd.DataFrame(rows)
st.dataframe(res, use_container_width=True, height=360)
st.download_button("Descargar resultados CSV", res.to_csv(index=False).encode("utf-8"), "resultados_v3.csv", "text/csv")

st.markdown("---"); st.markdown("### Fórmulas")
st.latex(r"P_{eje}=\frac{\rho g Q H}{\eta},\quad T_{\text{load}}=\frac{P_{eje}}{\omega}")
st.latex(r"J_{\text{eq}}=J_m + J_{\text{driver}} + r^2\,(J_{\text{imp}}+J_{\text{driven}}+J_{\text{fluido}})")
st.latex(r"\alpha(n)=\frac{T_{\text{avail}}-T_{\text{load}}(n)}{J_{\text{eq}}},\quad t=\int \frac{dn}{\min(\alpha\cdot 60/2\pi,\ \text{rampa}_{motor})}")
