
import streamlit as st
import pandas as pd
import numpy as np
import math
from io import StringIO
import matplotlib.pyplot as plt

st.set_page_config(page_title="Tiempo de reacción bombas con VDF", layout="wide")

# -------------------------------
# Helpers de cálculo
# -------------------------------
def inertia_disc_ring(mass_kg, D_mm):
    R = (D_mm/1000.0)/2.0
    J_disc = 0.5 * mass_kg * (R**2)
    J_ring = 1.0 * mass_kg * (R**2)
    return J_disc, J_ring

def fit_system_curve(points_df):
    """
    Ajusta H = H0 + K * Q^2.
    Acepta Q en m3/h o m3/s (detecta por magnitud).
    Devuelve H0 [m] y K_SI con Q en m3/s.
    """
    dfp = points_df.dropna()
    if {"Q","H"}.difference(dfp.columns):
        raise ValueError("Se requieren columnas 'Q' y 'H'.")
    Q = dfp["Q"].astype(float).to_numpy()
    H = dfp["H"].astype(float).to_numpy()
    if Q.max() > 200:  # umbral simple para detectar m3/h
        Qs = Q / 3600.0  # m3/s
    else:
        Qs = Q
    X = np.vstack([np.ones_like(Qs), Qs**2]).T  # [1, Q^2]
    beta, _, _, _ = np.linalg.lstsq(X, H, rcond=None)
    H0, K_SI = beta[0], beta[1]
    return float(H0), float(K_SI)

def tload_from_system(n_rpm, Qref_m3h, H0, K_SI, nref_rpm, eta=0.65, SG=1.0):
    """
    T_load(n) con H = H0 + K * Q^2 y Q ≈ alpha * n.
    Devuelve torque [N·m] equivalente en el eje del motor.
    """
    rho = 1000.0 * SG
    g = 9.81
    alpha = (Qref_m3h / 3600.0) / max(nref_rpm, 1e-6)  # (m3/s) / rpm
    A = rho * g * alpha * (60.0/(2.0*math.pi)) / max(eta,1e-6)  # N·m/m
    n = np.array(n_rpm, dtype=float)
    return A * (H0 + K_SI * (alpha**2) * (n**2))

def compute_results(df, motors_by_tag, ramp_motor_rpm_s=300.0, overload_pu=1.0,
                    inertia_model="range", sys_model_by_tag=None):
    rows = []
    for _, r in df.iterrows():
        tag = r["TAG"]
        m = motors_by_tag.get(tag, {})
        T_nom = float(m.get("T_nom", np.nan))
        Jm    = float(m.get("Jm", np.nan))
        if np.isnan(T_nom) or np.isnan(Jm):
            rows.append({"TAG": tag, "status":"Falta T_nom/Jm"})
            continue

        P_eff = float(r["P_eff [kW]"])
        n_motor_min = float(r["n_motor_min [rpm]"])
        n_motor_max = float(r["n_motor_max [rpm]"])
        n_ref = n_motor_max
        omega_ref = 2*math.pi*(n_ref/60.0)
        T_ref = (P_eff*1000.0)/max(omega_ref,1e-6)  # fallback afinidad

        trans_r = float(r["r"])
        D_mm = float(r["Ø_imp [mm]"])
        M_imp = float(r["M_imp [kg]"])
        Jp_disc, Jp_ring = inertia_disc_ring(M_imp, D_mm)
        J_eq_disc = Jm + (trans_r**2)*Jp_disc
        J_eq_ring = Jm + (trans_r**2)*Jp_ring

        pump_accel_vfd = ramp_motor_rpm_s / trans_r
        n_pump_min = float(r["n_pump_min [rpm]"])
        n_pump_max = float(r["n_pump_max [rpm]"])
        t_ramp_only = (n_pump_max - n_pump_min) / pump_accel_vfd if pump_accel_vfd>0 else np.inf

        # --- Carga: sistema si hay datos; si no, afinidad ---
        T_avail = overload_pu * T_nom
        use_sys = sys_model_by_tag and (tag in sys_model_by_tag)
        if use_sys:
            H0, K_SI, Qref_m3h, Href_m, eta, SG = sys_model_by_tag[tag]
            def Tload(nrpm): return tload_from_system(nrpm, Qref_m3h, H0, K_SI, n_ref, eta=eta, SG=SG)
        else:
            def Tload(nrpm): return T_ref * (nrpm/n_ref)**2

        def integrate_time(J_eq):
            t = 0.0
            steps = 600
            dn = (n_motor_max - n_motor_min)/steps
            for i in range(steps):
                n_mid = n_motor_min + (i+0.5)*dn
                T_load_mid = Tload(n_mid)
                torque_margin = T_avail - T_load_mid
                if torque_margin <= 0.0:
                    return np.inf
                alpha = torque_margin / J_eq  # rad/s^2
                accel_rpm_s_torque = (60.0/(2.0*math.pi))*alpha
                accel_rpm_s = min(accel_rpm_s_torque, ramp_motor_rpm_s)
                if accel_rpm_s <= 0:
                    return np.inf
                t += dn / accel_rpm_s
            return t

        t_disc = integrate_time(J_eq_disc)
        t_ring = integrate_time(J_eq_ring)

        if inertia_model == "disc":
            t_final = max(t_ramp_only, t_disc)
            rows.append({"TAG": tag, "modelo":"sistema" if use_sys else "afinidad",
                         "Pump accel VFD-only [rpm/s]": pump_accel_vfd,
                         "t_ramp_only [s]": t_ramp_only,
                         "t_disc [s]": t_disc, "t_final [s]": t_final})
        elif inertia_model == "ring":
            t_final = max(t_ramp_only, t_ring)
            rows.append({"TAG": tag, "modelo":"sistema" if use_sys else "afinidad",
                         "Pump accel VFD-only [rpm/s]": pump_accel_vfd,
                         "t_ramp_only [s]": t_ramp_only,
                         "t_ring [s]": t_ring, "t_final [s]": t_final})
        else:
            rows.append({"TAG": tag, "modelo":"sistema" if use_sys else "afinidad",
                         "Pump accel VFD-only [rpm/s]": pump_accel_vfd,
                         "t_ramp_only [s]": t_ramp_only,
                         "t_disc [s]": t_disc, "t_ring [s]": t_ring,
                         "t_final_min [s]": max(t_ramp_only, t_disc),
                         "t_final_max [s]": max(t_ramp_only, t_ring)})
    return pd.DataFrame(rows)

# ---------------------------------
# Datos por defecto (tabla + motor)
# ---------------------------------
default_table = pd.DataFrame([
    ("ROUGHER CONCENTRATE PUMP", "4210-PU-003", 37, 32.2, 4, 1475, 738, 3.15, 469, 234, 600.0, 228.7),
    ("REGRIND CYCLONE FEED PUMP", "4220-PU-010", 200, 173.9, 6, 990, 495, 2.47, 400, 200, 1000.0, 816.2),
    ("CLEANER 2 FEED PUMP", "4230-PU-011", 200, 173.9, 4, 1490, 745, 2.54, 588, 294, 750.0, 268.6),
    ("CLEANER SCAVENGER CONCENTRATE PUMP", "4230-PU-015", 75, 65.2, 4, 1485, 743, 3.44, 432, 216, 750.0, 268.6),
    ("CLEANER 2 TAILINGS PUMP", "4230-PU-022", 90, 78.3, 4, 1489, 745, 3.44, 433, 216, 750.0, 128.1),
    ("CLEANER 2 SPARGER FEED PUMP 1", "4230-PU-023", 90, 78.3, 4, 1489, 745, 2.42, 615, 308, 600.0, 228.7),
    ("CLEANER 2 SPARGER FEED PUMP 2", "4230-PU-024", 90, 78.3, 4, 1489, 745, 2.42, 615, 308, 600.0, 228.7),
    ("CLEANER 3 FEED PUMP", "4230-PU-031", 110, 95.7, 4, 1489, 745, 2.34, 635, 318, 600.0, 228.7),
], columns=["Application","TAG","P_inst [kW]","P_eff [kW]","Poles","n_motor_max [rpm]","n_motor_min [rpm]","r",
            "n_pump_max [rpm]","n_pump_min [rpm]","Ø_imp [mm]","M_imp [kg]"])

default_motors_by_tag = {
    "4210-PU-003": {"T_nom": 240.0, "Jm": 0.5177, "n_nom": 1475},
    "4220-PU-010": {"T_nom": 1930.0, "Jm": 11.0, "n_nom": 990},
    "4230-PU-011": {"T_nom": 1282.0, "Jm": 4.43, "n_nom": 1490},
    "4230-PU-015": {"T_nom": 482.0, "Jm": 1.64, "n_nom": 1485},
    "4230-PU-022": {"T_nom": 577.0, "Jm": 2.57, "n_nom": 1489},
    "4230-PU-023": {"T_nom": 577.0, "Jm": 2.57, "n_nom": 1489},
    "4230-PU-024": {"T_nom": 577.0, "Jm": 2.57, "n_nom": 1489},
    "4230-PU-031": {"T_nom": 706.0, "Jm": 2.57, "n_nom": 1489},
}

# -------------------------------
# Sidebar – parámetros de modelo
# -------------------------------
st.sidebar.header("Parámetros de cálculo")
ramp_motor = st.sidebar.number_input("Rampa VDF (rpm/s referidos al motor)", min_value=1.0, max_value=2000.0, value=300.0, step=10.0)
overload_pu = st.sidebar.number_input("Sobrecarga de par (pu)", min_value=0.5, max_value=2.5, value=1.0, step=0.1)
inertia_mode = st.sidebar.selectbox("Modelo de inercia del impulsor", options=["range", "disc", "ring"], index=0)

st.sidebar.header("Datos de motores (editable)")
motors_df = pd.DataFrame([{"TAG":k, **v} for k,v in default_motors_by_tag.items()])
motors_df = st.sidebar.data_editor(motors_df, num_rows="dynamic", use_container_width=True)
motors_by_tag = {row["TAG"]:{k:row[k] for k in row.index if k!="TAG"} for _, row in motors_df.iterrows()}

# -------------------------------
# Carga de datos
# -------------------------------
st.title("Tiempo de reacción de bombas con VDF – MantoVerde")
st.caption("App de modelación (screening/modelo de sistema) para estimar tiempos de aceleración/deceleración.")

uploaded = st.file_uploader("Carga una tabla CSV con las columnas mínimas (ver ejemplo). Si no, se usará el dataset por defecto.", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = default_table.copy()

st.subheader("Tabla de entradas (editable)")
df = st.data_editor(df, use_container_width=True, num_rows="dynamic")
st.download_button("Descargar entradas CSV", df.to_csv(index=False).encode("utf-8"), "entradas_bombas.csv", "text/csv")

# Calcular r desde poleas si existen
with st.expander("Opcional: calcular r desde poleas"):
    st.write("Si tu tabla tiene columnas de poleas, puedo calcular r automáticamente.")
    st.caption("Columnas esperadas: 'DriveRSheave_in'/'DriveNSheave_in' o 'DriveRSheave_inch'/'DriveNSheave_inch'")
    usar_r_auto = st.checkbox("Usar r calculada con poleas si están disponibles", value=True)
    slip = st.number_input("Deslizamiento de correas (%)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
    posibles = [("DriveRSheave_in","DriveNSheave_in"),
                ("DriveRSheave_inch","DriveNSheave_inch")]
    r_auto_col = None
    for c_drv, c_drn in posibles:
        if c_drv in df.columns and c_drn in df.columns:
            r_auto_col = (c_drv, c_drn)
            break
    if r_auto_col is not None:
        c_drv, c_drn = r_auto_col
        try:
            df["r_auto"] = df[c_drn].astype(float) / df[c_drv].astype(float)
            df["r_auto"] = df["r_auto"] * (1.0/(1.0 - slip/100.0))  # ajuste por slip
            st.dataframe(df[["TAG","r","r_auto"]])
            if usar_r_auto:
                df["r"] = df["r_auto"]
        except Exception as e:
            st.warning(f"No se pudo calcular r_auto: {e}")
    else:
        st.info("No se encontraron columnas de poleas. Mantengo r de la tabla.")

# -------------------------------
# Curvas de sistema (opcional)
# -------------------------------
with st.expander("Curvas de sistema por TAG (opcional)"):
    st.write("Pega puntos Q–H para ajustar H = H0 + K·Q² (puedes usar m³/h o m³/s).")
    st.caption("Formato mínimo: columnas 'TAG','Q','H'. Puedes mezclar varios TAGs en la misma tabla.")
    ejemplo = pd.DataFrame({
        "TAG": ["4210-PU-003"]*3 + ["4230-PU-011"]*3,
        "Q":   [200,400,600,  300,600,900],  # m3/h
        "H":   [6.0,9.5,15.0, 10.0,16.0,25.0] # m
    })
    sys_points = pd.DataFrame(ejemplo)
    sys_points = st.data_editor(sys_points, use_container_width=True, num_rows="dynamic")
    st.download_button("Descargar ejemplo CSV", sys_points.to_csv(index=False).encode("utf-8"),
                       "syscurve_points.csv", "text/csv")

    st.write("Define duty por TAG (Q_ref,H_ref,eta,SG):")
    duty_demo = pd.DataFrame({
        "TAG": df["TAG"],
        "Q_ref_m3h": [500]*len(df),
        "H_ref_m":   [12]*len(df),
        "eta":       [0.65]*len(df),
        "SG":        [1.0]*len(df)
    })
    duty_tbl = st.data_editor(duty_demo, use_container_width=True, num_rows="dynamic")

# -------------------------------
# Enfoque con LaTeX
# -------------------------------
st.markdown("### Enfoque seguido")
st.latex(r"Q \propto n,\quad H \propto n^{2},\quad P \propto n^{3}\ \Rightarrow\ T_{\mathrm{load}} \propto n^{2}")
st.latex(r"J_{\mathrm{eq}} = J_m + r^{2}\,J_p")
st.latex(r"J_p^{\text{disco}}=\tfrac{1}{2} m R^{2},\qquad J_p^{\text{aro}}= m R^{2}")
st.latex(r"a_{\text{VDF,bomba}}=\dfrac{\text{rampa}_{\text{motor}}}{r}")
st.latex(r"a_{\text{par}}=\dfrac{T_{\text{avail}} - T_{\text{load}}(n)}{J_{\text{eq}}}")
st.latex(r"t_{\text{final}}=\max\{\,t_{\text{rampa}},\ t_{\text{par/inercia}}\,\}")

st.markdown("### Supuestos")
st.markdown("""
- Zona de **par constante** hasta vel. nominal (sin debilitamiento de campo).  
- **Sobrecarga de par** configurable (pu); límite real puede ser por VDF/corriente.  
- Por ahora no se incluyen inercias de **acoples/poleas** ni **curva del sistema** salvo si se ingresa arriba.  
- En **deceleración** podría requerirse **freno dinámico** para cumplir tiempos sin sobrevoltaje del bus DC.
""")

# -------------------------------
# Cálculo - construir sys_model_by_tag a partir de editores
# -------------------------------
sys_model_by_tag = {}
if len(sys_points) > 1 and "TAG" in sys_points.columns:
    for tag in sys_points["TAG"].unique():
        pts = sys_points[sys_points["TAG"]==tag][["Q","H"]]
        if len(pts) >= 2:
            try:
                H0, K_SI = fit_system_curve(pts)
                duty_row = duty_tbl[duty_tbl["TAG"]==tag]
                if not duty_row.empty:
                    Qref = float(duty_row["Q_ref_m3h"].iloc[0])
                    Href = float(duty_row["H_ref_m"].iloc[0])
                    eta  = float(duty_row["eta"].iloc[0])
                    SG   = float(duty_row["SG"].iloc[0])
                    sys_model_by_tag[tag] = (H0, K_SI, Qref, Href, eta, SG)
            except Exception as e:
                st.warning(f"No se pudo ajustar curva de sistema para {tag}: {e}")

# -------------------------------
# Resultados
# -------------------------------
st.subheader("Resultados")
res = compute_results(df, motors_by_tag, ramp_motor_rpm_s=ramp_motor,
                      overload_pu=overload_pu, inertia_model=inertia_mode,
                      sys_model_by_tag=sys_model_by_tag)

st.dataframe(res, use_container_width=True, height=350)
st.download_button("Descargar resultados CSV", res.to_csv(index=False).encode("utf-8"), "resultados_tiempos.csv", "text/csv")

# -------------------------------
# Gráfico por TAG (rampa VDF ideal)
# -------------------------------
st.subheader("Gráfico Δn vs t por rampa VDF (ideal)")
tag_sel = st.selectbox("Selecciona un TAG", options=df["TAG"].tolist())
r_sel = float(df.loc[df["TAG"]==tag_sel, "r"].iloc[0])
pump_accel = ramp_motor / r_sel if r_sel>0 else 0.0
n1 = float(df.loc[df["TAG"]==tag_sel, "n_pump_min [rpm]"].iloc[0])
n2 = float(df.loc[df["TAG"]==tag_sel, "n_pump_max [rpm]"].iloc[0])
t_ramp_only = (n2-n1)/pump_accel if pump_accel>0 else np.inf
t_vals = np.linspace(0, t_ramp_only if np.isfinite(t_ramp_only) else 1.0, 100)
n_vals = n1 + pump_accel*t_vals

fig = plt.figure()
plt.plot(t_vals, n_vals)
plt.xlabel("Tiempo [s]")
plt.ylabel("n bomba [rpm]")
plt.title(f"{tag_sel} – Rampa VDF: {pump_accel:.1f} rpm/s (bomba)")
st.pyplot(fig)

# -------------------------------
# Conclusiones y mejoras
# -------------------------------
st.markdown("### Conclusiones preliminares")
st.markdown("""
- El mínimo teórico lo acota la **rampa del VDF**.  
- Si el **par disponible** es bajo y/o \\(J_{eq}\\) alto, manda la **dinámica por par/inercia**.  
- Recomendado definir **rampas diferenciadas** (subida/bajada) y validar **corriente** y **NPSH** para puntos acelerados.
""")

st.markdown("### Información adicional recomendada")
st.markdown("""
- **Curva bomba + curva del sistema** (TDH, SG, FF) para estimar \\(T_{load}(n)\\) real.  
- **Límites de corriente y sobrecarga temporal** del VDF por TAG.  
- **Inercia de transmisión** (poleas/acoples) o **inercia de impulsor** del fabricante.
""")
