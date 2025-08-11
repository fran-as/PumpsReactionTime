
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

def integrate_time(n1_motor, n2_motor, r, J_eq, T_avail, T_ref, n_ref, steps=400, ramp_motor_rpm_s=300.0):
    # Integra tiempo de n1->n2 (rpm motor) con límite por par e inercia + rampa del VDF
    t = 0.0
    dn = (n2_motor - n1_motor)/steps
    for i in range(steps):
        n_mid = n1_motor + (i+0.5)*dn
        # Torque de carga ~ n^2 (normalizado en n_ref con T_ref)
        T_load = T_ref * (n_mid/n_ref)**2
        torque_margin = T_avail - T_load
        if torque_margin <= 0.0:
            return np.inf
        alpha = torque_margin / J_eq  # rad/s^2 en el eje del motor
        accel_rpm_s_torque = (60.0/(2*math.pi))*alpha
        accel_rpm_s = min(accel_rpm_s_torque, ramp_motor_rpm_s)
        if accel_rpm_s <= 0:
            return np.inf
        t += dn / accel_rpm_s
    return t

def compute_results(df, motors_by_tag, ramp_motor_rpm_s=300.0, overload_pu=1.0, inertia_model="range"):
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
        T_ref = (P_eff*1000.0)/omega_ref  # torque consumido a n_ref

        trans_r = float(r["r"])
        D_mm = float(r["Ø_imp [mm]"])
        M_imp = float(r["M_imp [kg]"])
        Jp_disc, Jp_ring = inertia_disc_ring(M_imp, D_mm)

        J_eq_disc = Jm + (trans_r**2)*Jp_disc
        J_eq_ring = Jm + (trans_r**2)*Jp_ring

        # Rampa bomba sólo por VDF
        pump_accel_vfd = ramp_motor_rpm_s / trans_r
        n_pump_min = float(r["n_pump_min [rpm]"])
        n_pump_max = float(r["n_pump_max [rpm]"])
        t_ramp_only = (n_pump_max - n_pump_min) / pump_accel_vfd if pump_accel_vfd>0 else np.inf

        T_avail = overload_pu * T_nom

        t_disc = integrate_time(n_motor_min, n_motor_max, trans_r, J_eq_disc, T_avail, T_ref, n_ref, ramp_motor_rpm_s=ramp_motor_rpm_s)
        t_ring = integrate_time(n_motor_min, n_motor_max, trans_r, J_eq_ring, T_avail, T_ref, n_ref, ramp_motor_rpm_s=ramp_motor_rpm_s)

        if inertia_model == "disc":
            t_final = max(t_ramp_only, t_disc)
        elif inertia_model == "ring":
            t_final = max(t_ramp_only, t_ring)
        else:
            # range: devolvemos ambos y el rango
            t_final = max(t_ramp_only, t_disc)
            t_final_max = max(t_ramp_only, t_ring)

        row = {
            "TAG": tag,
            "Pump accel VFD-only [rpm/s]": pump_accel_vfd,
            "t_ramp_only [s]": t_ramp_only,
            "t_disc [s]": t_disc,
            "t_ring [s]": t_ring,
        }
        if inertia_model == "range":
            row["t_final_min [s]"] = t_final
            row["t_final_max [s]"] = t_final_max
        else:
            row["t_final [s]"] = t_final
        rows.append(row)

    res = pd.DataFrame(rows)
    return res

# ---------------------------------
# Datos por defecto (tabla + motor)
# ---------------------------------
default_table = pd.DataFrame([
    # Application, TAG, P_inst, P_eff, Poles, n_motor_max, n_motor_min, r, n_pump_max, n_pump_min, Ø_imp, M_imp
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
st.caption("App de modelación (screening) para estimar tiempos de aceleración/deceleración por rampa VDF y por par/inercia.")

uploaded = st.file_uploader("Carga una tabla CSV con las columnas mínimas (ver ejemplo). Si no, se usará el dataset por defecto.", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = default_table.copy()

st.subheader("Tabla de entradas (editable)")
df = st.data_editor(df, use_container_width=True, num_rows="dynamic")
st.download_button("Descargar entradas CSV", df.to_csv(index=False).encode("utf-8"), "entradas_bombas.csv", "text/csv")

# -------------------------------
# Cálculo
# -------------------------------
st.subheader("Resultados")
res = compute_results(df, motors_by_tag, ramp_motor_rpm_s=ramp_motor, overload_pu=overload_pu, inertia_model=inertia_mode)

st.dataframe(res, use_container_width=True, height=350)
st.download_button("Descargar resultados CSV", res.to_csv(index=False).encode("utf-8"), "resultados_tiempos.csv", "text/csv")

# -------------------------------
# Gráfico por TAG (opcional)
# -------------------------------
st.subheader("Gráfico de aceleración ideal (por rampa VDF) – selección rápida")
tag_sel = st.selectbox("Selecciona un TAG para visualizar Δn_bomba vs tiempo por rampa VDF", options=df["TAG"].tolist())
r_sel = float(df.loc[df["TAG"]==tag_sel, "r"].iloc[0])
pump_accel = ramp_motor / r_sel
n1 = float(df.loc[df["TAG"]==tag_sel, "n_pump_min [rpm]"].iloc[0])
n2 = float(df.loc[df["TAG"]==tag_sel, "n_pump_max [rpm]"].iloc[0])
t_ramp_only = (n2-n1)/pump_accel if pump_accel>0 else np.inf
t_vals = np.linspace(0, t_ramp_only, 100)
n_vals = n1 + pump_accel*t_vals

fig = plt.figure()
plt.plot(t_vals, n_vals)
plt.xlabel("Tiempo [s]")
plt.ylabel("n bomba [rpm]")
plt.title(f"{tag_sel} – Rampa VDF: {pump_accel:.1f} rpm/s (bomba)")
st.pyplot(fig)

# -------------------------------
# Secciones de memoria: enfoque y supuestos
# -------------------------------
st.markdown("""
### Enfoque seguido
1. **Afinidad** (screening): \\(Q\\propto n\\), \\(H\\propto n^2\\), \\(P\\propto n^3\\) ⇒ \\(T_{load}\\propto n^2\\).
2. **Inercia equivalente** al eje motor: \\(J_{eq}=J_m+r^2 J_p\\). \\(J_p\\) se acota entre **disco** (\\(J=\\tfrac{1}{2}mR^2\\)) y **aro** (\\(J=mR^2\\)).
3. **Rampa del VDF**: aceleración de bomba = **rampa_motor / r**.
4. **Tiempo por par/inercia**: \\(a_{par}=[(T_{avail}-T_{load})/J_{eq}]\\) ⇢ integración en velocidad.
5. **Tiempo final** por TAG: máximo entre **tiempo por rampa** y **tiempo por par/inercia**.
""")

st.markdown("""
### Supuestos
- Zona de **par constante** hasta velocidad nominal del motor (sin debilitamiento de campo).
- **Sobrecarga de par** configurable (pu). Límite real puede estar fijado por el VDF/corriente.
- **No** se incluyen (por ahora) inercias de poleas/acoples ni **curva del sistema** (TDH/SG/FF); la carga se aproxima por afinidad.
- Deceleración puede requerir **freno dinámico** para cumplir tiempos sin sobrevoltaje.
""")

st.markdown("""
### Conclusiones preliminares
- El tiempo mínimo absoluto está **acotado por la rampa** del VDF: \\(t=\\Delta n_{bomba}/(\\text{rampa}/r)\\).
- Si el **par disponible** (en pu) es bajo y/o \\(J_{eq}\\) es alto, el tiempo **real** lo domina la **dinámica por par/inercia**.
- Es recomendable definir **rampas diferenciadas** por TAG (subida/bajada) y validar **corriente** y **NPSH** para los puntos acelerados.
""")

st.markdown("""
### Información adicional recomendada
- **Curva bomba + curva del sistema** por TAG para estimar \\(T_{load}(n)\\) real.
- **Límites de corriente y tiempo de sobrecarga** del VDF por TAG (1.0–1.5 pu típicamente).
- **Inercia de transmisión** (poleas, acoples) y/o **inercia real del impulsor** si está disponible del fabricante.
- **Volumen y setpoints** de cada cajón para traducir \\(\\Delta Q\\) en **respuesta de nivel** (impacto en control).
""")
