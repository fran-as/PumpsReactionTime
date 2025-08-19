# -*- coding: utf-8 -*-
"""
Memoria de Cálculo – Tiempo de reacción de bombas (VDF)
Colocar en la raíz: bombas_dataset_with_torque_params.xlsx
"""

import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ===========================
# Utilidades
# ===========================
def norm(s: str) -> str:
    return (str(s).strip().lower()
            .replace(" ", "_").replace("-", "_")
            .replace("(", "").replace(")", "").replace("%", "pct"))

def pick(cols, *candidates, default=None):
    for c in candidates:
        if c in cols:
            return c
    return default

def to_float(x):
    try:
        if pd.isna(x):
            return np.nan
        s = str(x).strip()
        if s == "":
            return np.nan
        if "," in s and "." in s:
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", ".")
        return float(s)
    except Exception:
        try:
            return float(x)
        except Exception:
            return np.nan

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def fmt2(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "—"
        return f"{float(x):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(x)

def lnum(x, nd=2):
    """Número para LaTeX con punto decimal (evita NaN)."""
    try:
        xv = float(x)
        if np.isnan(xv) or np.isinf(xv):
            return r"\mathrm{NaN}"
        return f"{xv:.{nd}f}"
    except Exception:
        return r"\mathrm{NaN}"

def badge(value, unit=""):
    return f"<span style='background:#1f6c4e;color:white;border-radius:8px;padding:2px 8px;font-weight:600;'>{fmt2(value)}{(' '+unit) if unit else ''}</span>"

def line_value(label, value, unit=""):
    st.markdown(f"**{label}:** {badge(value, unit)}", unsafe_allow_html=True)


# ===========================
# Datos
# ===========================
DATA_PATH = "bombas_dataset_with_torque_params.xlsx"
raw = pd.read_excel(DATA_PATH)
df = raw.copy()
df.columns = [norm(c) for c in df.columns]
cols = set(df.columns)

# a numérico (SI)
for c in ["driverpulley_j_kgm2","driverbushing_j_kgm2","drivenpulley_j_kgm2","drivenbushing_j_kgm2"]:
    if c in df.columns and df[c].dtype == object:
        df[c] = df[c].apply(to_float)

if "j_driver_total_kgm2" not in df.columns: df["j_driver_total_kgm2"] = np.nan
if "j_driven_total_kgm2" not in df.columns: df["j_driven_total_kgm2"] = np.nan

# mapeo columnas
col_tag      = df.columns[0]
col_power_kw = pick(cols, "motorpower_kw","motorpowerefective_kw","motorpowerinstalled_kw","power_kw")
col_poles    = pick(cols, "poles")
col_r        = pick(cols, "r_trans","relaciontransmision","relacion_transmision")
col_nmot_min = pick(cols, "motorspeedmin_rpm","motor_n_min_rpm","n_motor_min")
col_nmot_max = pick(cols, "motorspeedmax_rpm","motor_n_max_rpm","n_motor_max")
col_npum_min = pick(cols, "pump_n_min_rpm","n_pump_min")
col_npum_max = pick(cols, "pump_n_max_rpm","n_pump_max")
col_jm       = pick(cols, "motor_j_kgm2","j_motor_kgm2")
col_jdrv_tot = "j_driver_total_kgm2"
col_jdrn_tot = "j_driven_total_kgm2"
col_jimp     = pick(cols, "impeller_j_kgm2","j_impeller_kgm2","j_impulsor_kgm2")
col_dimp     = pick(cols, "impeller_d_mm","diametroimpulsor_mm")
col_tnom     = pick(cols, "t_nom_nm")

# hidráulica
col_H0     = pick(cols, "h0_m")
col_K      = pick(cols, "k_m_s2")
col_eta_a  = pick(cols, "eta_a")
col_eta_b  = pick(cols, "eta_b")
col_eta_c  = pick(cols, "eta_c")
col_rho    = pick(cols, "rho_kgm3")
col_nref   = pick(cols, "n_ref_rpm")
col_Qmin_h = pick(cols, "q_min_m3h")
col_Qmax_h = pick(cols, "q_max_m3h")

for c in [col_power_kw, col_poles, col_r, col_nmot_min, col_nmot_max, col_npum_min, col_npum_max,
          col_jm, col_jdrv_tot, col_jdrn_tot, col_jimp, col_dimp, col_tnom,
          col_H0, col_K, col_eta_a, col_eta_b, col_eta_c, col_rho, col_nref, col_Qmin_h, col_Qmax_h]:
    if c and df[c].dtype == object:
        df[c] = df[c].apply(to_float)


# ===========================
# UI
# ===========================
st.set_page_config(page_title="Memoria de Cálculo – Bombas (VDF)", layout="wide")
st.title("Memoria de Cálculo – Tiempo de reacción de bombas (VDF)")

tags = sorted(df.iloc[:, 0].astype(str).unique().tolist())
tag = st.sidebar.selectbox("TAG", tags)

st.sidebar.header("Parámetros de simulación")
ramp_rpm_s = st.sidebar.number_input("Rampa VDF [rpm/s] (eje motor)", min_value=1.0, value=300.0, step=10.0, format="%.2f")
dt = st.sidebar.number_input("Paso de integración dt [s] (hidráulica)", min_value=0.001, value=0.01, step=0.01, format="%.3f")

row = df.loc[df.iloc[:,0].astype(str)==tag].iloc[0]


# ===========================
# 1) Entrada
# ===========================
st.header("1) Parámetros de entrada (dato)")

c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("Motor / Transmisión")
    line_value("Potencia motor", row.get(col_power_kw, np.nan), "kW")
    if col_poles:
        pol = row.get(col_poles)
        st.markdown(f"**Polos:** {badge(int(pol)) if not pd.isna(pol) else '—'}", unsafe_allow_html=True)
    line_value("Relación r = ω_motor/ω_bomba", row.get(col_r, np.nan))
    st.markdown(f"**Velocidad motor min–max [rpm]:** {badge(row.get(col_nmot_min, np.nan))} – {badge(row.get(col_nmot_max, np.nan))}", unsafe_allow_html=True)
    if (col_npum_min in df.columns) and (col_npum_max in df.columns):
        st.markdown(f"**Velocidad bomba min–max [rpm]:** {badge(row.get(col_npum_min, np.nan))} – {badge(row.get(col_npum_max, np.nan))}", unsafe_allow_html=True)

with c2:
    st.subheader("Inercias (kg·m²)")
    drv_p = float(row.get("driverpulley_j_kgm2", np.nan)) if "driverpulley_j_kgm2" in df.columns else np.nan
    drv_b = float(row.get("driverbushing_j_kgm2", np.nan)) if "driverbushing_j_kgm2" in df.columns else np.nan
    j_driver = np.nansum([drv_p, drv_b])

    drn_p = float(row.get("drivenpulley_j_kgm2", np.nan)) if "drivenpulley_j_kgm2" in df.columns else np.nan
    drn_b = float(row.get("drivenbushing_j_kgm2", np.nan)) if "drivenbushing_j_kgm2" in df.columns else np.nan
    j_driven = np.nansum([drn_p, drn_b])

    line_value("J_m (motor)", row.get(col_jm, np.nan))
    st.markdown(f"**J_driver (total):** {badge(j_driver)}  \n└ = polea({fmt2(drv_p)}) + manguito({fmt2(drv_b)})", unsafe_allow_html=True)
    st.markdown(f"**J_driven (total):** {badge(j_driven)}  \n└ = polea({fmt2(drn_p)}) + manguito({fmt2(drn_b)})", unsafe_allow_html=True)
    line_value("J_imp (impulsor)", row.get(col_jimp, np.nan))

with c3:
    st.subheader("Impulsor / Hidráulica")
    if col_dimp: line_value("Diámetro impulsor", row.get(col_dimp, np.nan), "mm")
    line_value("H0", row.get(col_H0, np.nan), "m")
    line_value("K", row.get(col_K, np.nan), "m·s⁻²")
    st.write("**η(Q) = a + bQ + cQ²** (Q en m³/s)")
    st.markdown(f"a={badge(row.get(col_eta_a, np.nan))}, b={badge(row.get(col_eta_b, np.nan))}, c={badge(row.get(col_eta_c, np.nan))}", unsafe_allow_html=True)
    line_value("ρ", row.get(col_rho, np.nan), "kg/m³")
    line_value("n_ref", row.get(col_nref, np.nan), "rpm")
    st.markdown(f"**Rango Q (m³/h):** {badge(row.get(col_Qmin_h, np.nan))} – {badge(row.get(col_Qmax_h, np.nan))}", unsafe_allow_html=True)


# ===========================
# 2) Inercia equivalente
# ===========================
st.header("2) Inercia equivalente al eje del motor")
st.latex(r"J_{\mathrm{eq}} \;=\; J_m \;+\; J_{\mathrm{driver}} \;+\; \dfrac{J_{\mathrm{driven}} + J_{\mathrm{imp}}}{r^{2}}")

with st.popover("ⓘ ¿Por qué dividir por r²?"):
    st.markdown(
        r"Las inercias del **lado bomba** giran a \(\omega_p=\omega_m/r\). "
        r"Igualando energías cinéticas y evaluando todo a la misma \(\omega_m\):"
    )
    st.latex(
        r"\tfrac12 J_{eq}\,\omega_m^2 \;=\; "
        r"\tfrac12 J_m\,\omega_m^2 \;+\; \tfrac12 J_{driver}\,\omega_m^2 \;+\; "
        r"\tfrac12 J_{driven}\,\omega_p^2 \;+\; \tfrac12 J_{imp}\,\omega_p^2"
    )
    st.markdown(r"Sustituyendo \(\omega_p=\omega_m/r\) y dividiendo por \(\tfrac12\,\omega_m^2\):")
    st.latex(r"J_{eq}=J_m+J_{driver}+\dfrac{J_{driven}+J_{imp}}{r^2}")

J_m   = float(row.get(col_jm, 0.0) or 0.0)
J_drv = float(j_driver if not np.isnan(j_driver) else 0.0)
J_drn = float(j_driven if not np.isnan(j_driven) else 0.0)
J_imp = float(row.get(col_jimp, 0.0) or 0.0)
r_tr  = float(row.get(col_r, np.nan) or np.nan)

J_ref = (J_drn + J_imp) / (r_tr**2) if (not pd.isna(r_tr) and r_tr != 0) else np.nan
J_eq  = (J_m + J_drv + J_ref) if not np.isnan(J_ref) else np.nan

st.latex(
    rf"J_{{eq}}={lnum(J_m)}+{lnum(J_drv)}+\frac{{{lnum(J_drn)}+{lnum(J_imp)}}}{{{lnum(r_tr)}^2}}"
    rf"={lnum(J_eq)}\ \mathrm{{kg\,m^2}}"
)

contrib = pd.DataFrame([
    {"Componente": "Lado motor (J_m + J_driver)", "kg·m²": round(J_m + J_drv, 4)},
    {"Componente": "Reflejado lado bomba ((J_driven + J_imp)/r²)", "kg·m²": round(J_ref, 4) if not np.isnan(J_ref) else np.nan},
    {"Componente": "Total J_eq", "kg·m²": round(J_eq, 4) if not np.isnan(J_eq) else np.nan},
])
st.dataframe(contrib, use_container_width=True)


# ===========================
# 3) Tiempo sin hidráulica
# ===========================
st.header("3) Tiempo de reacción **sin** hidráulica")
st.latex(r"\dot n_{\mathrm{torque}}=\dfrac{60}{2\pi}\,\dfrac{T_{\mathrm{nom}}}{J_{\mathrm{eq}}}")
st.latex(r"t_{\mathrm{par}}=\dfrac{\Delta n}{\dot n_{\mathrm{torque}}},\qquad "
         r"t_{\mathrm{rampa}}=\dfrac{\Delta n}{\mathrm{rampa}_{\mathrm{VDF}}},\qquad "
         r"t_{\mathrm{final,sin}}=\max\!\left(t_{\mathrm{par}},\,t_{\mathrm{rampa}}\right)")

with st.popover("ⓘ Ayuda"):
    st.markdown(
        r"""- **Δn**: salto de velocidad del motor (rpm).
- **t_par** y **t_rampa** se calculan con:"""
    )
    st.latex(r"\dot n_{\mathrm{torque}}=\dfrac{60}{2\pi}\,\dfrac{T_{\mathrm{nom}}}{J_{\mathrm{eq}}}")
    st.latex(r"t_{\mathrm{par}}=\dfrac{\Delta n}{\dot n_{\mathrm{torque}}},\qquad "
             r"t_{\mathrm{rampa}}=\dfrac{\Delta n}{\mathrm{rampa}_{\mathrm{VDF}}}")
    st.latex(r"t_{\mathrm{final,sin}}=\max\!\big(t_{\mathrm{par}},\,t_{\mathrm{rampa}}\big)")
    st.markdown(
        r"**Interpretación**: si \(t_{par}>t_{rampa}\) la limitación es **par/inercia**; "
        r"si \(t_{rampa}>t_{par}\) manda la **rampa del VDF**."
    )

nmin = float(row.get(col_nmot_min, 0.0) or 0.0)
nmax = float(row.get(col_nmot_max, 0.0) or 0.0)
if col_tnom and not pd.isna(row.get(col_tnom)):
    T_nom = float(row[col_tnom])
else:
    P_kw = float(row.get(col_power_kw, 0.0) or 0.0)
    n_for_t = float(row.get(col_nmot_max, nmax) or nmax or 1.0)
    T_nom = 9550.0 * P_kw / max(n_for_t, 1.0)

m1, m2, m3 = st.columns(3)
m1.metric("Velocidad motor inicial [rpm]", fmt2(nmin))
m2.metric("Velocidad motor final [rpm]", fmt2(nmax))
m3.metric("Par disponible T_nom [Nm]", fmt2(T_nom))

delta_n = max(0.0, nmax - nmin)
accel_torque = (60.0/(2.0*math.pi)) * (T_nom / max(J_eq, 1e-9))  # rpm/s
t_par   = delta_n / max(accel_torque, 1e-9)
t_rampa = delta_n / max(ramp_rpm_s, 1e-9)
t_nohyd = max(t_par, t_rampa)

# Resumen explícito en LaTeX
st.latex(
    rf"\Delta n = {lnum(delta_n)}\ \mathrm{{rpm}},\quad "
    rf"\dot n_{{\mathrm{{torque}}}} = {lnum(accel_torque)}\ \mathrm{{rpm/s}},\quad "
    rf"t_{{par}} = {lnum(t_par)}\ \mathrm{{s}},\quad "
    rf"t_{{rampa}} = {lnum(t_rampa)}\ \mathrm{{s}},\quad "
    rf"t_{{final,sin}} = \max({lnum(t_par)},\,{lnum(t_rampa)}) = {lnum(t_nohyd)}\ \mathrm{{s}}"
)

if t_par > t_rampa:
    dominante = "Limitación por **par/inercia** (el VDF permite más rampa que el par disponible)."
    color_box = st.warning
else:
    dominante = "Limitación por **rampa del VDF** (el par alcanzaría más rápido que la rampa configurada)."
    color_box = st.info
color_box(dominante)

# “Qué pasaría si”
rampa_necesaria = delta_n / 1.0
capacidad_por_par = accel_torque
st.markdown(
    f"**Para Δn en 1,00 s:** rampa requerida = {badge(rampa_necesaria, 'rpm/s')}  \n"
    f"**Capacidad por par:** {badge(capacidad_por_par, 'rpm/s')}  \n"
    f"**Rampa VDF actual:** {badge(ramp_rpm_s, 'rpm/s')}",
    unsafe_allow_html=True
)


# ===========================
# 4) Hidráulica
# ===========================
st.header("4) Curva de sistema y tiempo **con** hidráulica")
st.latex(r"H(Q)=H_0+K\,Q^2,\qquad \eta(Q)=a+bQ+cQ^2")
st.latex(r"T_{\mathrm{pump}}(Q,n)=\dfrac{\rho\,g\,Q\,[H_0+KQ^2]}{\eta(Q)\,\omega(n)},\quad "
         r"\omega(n)=\dfrac{2\pi n}{60},\qquad T_{\mathrm{motor}}=\dfrac{T_{\mathrm{pump}}}{r}")
st.latex(r"\dfrac{dn_{\mathrm{motor}}}{dt}=\min\!\Big(\mathrm{rampa}_{\mathrm{VDF}},\; "
         r"\dfrac{60}{2\pi}\dfrac{T_{\mathrm{nom}}-T_{\mathrm{motor}}(n)}{J_{\mathrm{eq}}}\Big)")

H0     = float(row.get(col_H0, np.nan) or np.nan)
K      = float(row.get(col_K,  np.nan) or np.nan)
eta_a  = float(row.get(col_eta_a, 0.0) or 0.0)
eta_b  = float(row.get(col_eta_b, 0.0) or 0.0)
eta_c  = float(row.get(col_eta_c, 0.0) or 0.0)
rho    = float(row.get(col_rho, 1000.0) or 1000.0)

def eta_of_Q(Q_m3s): return eta_a + eta_b*Q_m3s + eta_c*(Q_m3s**2)
def H_of_Q(Q_m3s):   return H0 + K*(Q_m3s**2)

n_ref  = float(row.get(col_nref, nmax) or nmax or 1.0)
Qmin_h = float(row.get(col_Qmin_h, 0.0) or 0.0)
Qmax_h = float(row.get(col_Qmax_h, 0.0) or 0.0)
Qref_h = (Qmin_h + Qmax_h)/2.0 if (Qmax_h and Qmax_h > 0) else 0.0

alpha_default = (Qref_h / max(n_ref, 1.0))  # m³/h por rpm bomba
alpha_user = st.number_input("α: caudal por rpm de bomba [m³/h·rpm]", value=float(alpha_default), step=1.0, format="%.2f")
alpha = alpha_user / 3600.0                  # → m³/s por rpm

# Curva H(Q)
Qh = np.linspace(max(Qmin_h*0.5, 1.0), max(Qmax_h*1.2, max(50.0, Qref_h*1.5)), 120)
figH = go.Figure()
figH.add_trace(go.Scatter(x=Qh, y=H_of_Q(Qh/3600.0), mode="lines", name="H(Q)=H0+KQ²", line=dict(width=3)))
figH.update_layout(xaxis_title="Q [m³/h]", yaxis_title="H [m]", template="plotly_white", height=400)
st.plotly_chart(figH, use_container_width=True)

def simulate_with_hyd(n0, n1, dt, T_nom, J_eq, r_tr, alpha_m3s_per_rpm,
                      H0, K, eta_a, eta_b, eta_c, rho, ramp_rpm_s):
    g = 9.80665
    t, n = 0.0, n0
    times, n_list, q_list = [0.0], [n0], [0.0]
    max_steps = int(1e6)
    steps = 0
    while n < n1 and steps < max_steps:
        steps += 1
        n_cmd = min(n1, n + ramp_rpm_s*dt)
        n_b   = n / max(r_tr, 1e-9)
        Q     = max(0.0, alpha_m3s_per_rpm * n_b)   # m3/s
        eta   = clamp(eta_of_Q(Q), 0.3, 0.9)
        H     = H_of_Q(Q)
        omega_b = 2*math.pi*max(n_b,1e-6)/60
        P_h   = rho * g * Q * H
        T_pump = (P_h / max(eta, 1e-6)) / omega_b
        T_mot_load = T_pump / max(r_tr, 1e-9)

        accel_torque = (60/(2*math.pi)) * ((T_nom - T_mot_load)/max(J_eq,1e-9))
        accel = min(ramp_rpm_s, max(accel_torque, 0.0))
        n = min(n_cmd, n + accel*dt)
        t += dt

        times.append(t); n_list.append(n); q_list.append(Q*3600.0)
        if n >= n1 - 1e-6: break

    return pd.DataFrame({"t_s": times, "n_motor_rpm": n_list, "Q_m3h": q_list})

if not any(pd.isna([J_eq, r_tr, H0, K, rho])):
    sim = simulate_with_hyd(
        n0=nmin, n1=nmax, dt=dt, T_nom=T_nom, J_eq=J_eq, r_tr=r_tr,
        alpha_m3s_per_rpm=alpha, H0=H0, K=K,
        eta_a=eta_a, eta_b=eta_b, eta_c=eta_c, rho=rho, ramp_rpm_s=ramp_rpm_s
    )
    t_hyd = float(sim["t_s"].iloc[-1])

    figN = go.Figure()
    figN.add_trace(go.Scatter(x=sim["t_s"], y=sim["n_motor_rpm"], mode="lines", name="n_motor(t)"))
    figN.update_layout(template="plotly_white", xaxis_title="t [s]", yaxis_title="n_motor [rpm]", height=360)
    st.plotly_chart(figN, use_container_width=True)

    figQ = go.Figure()
    figQ.add_trace(go.Scatter(x=sim["t_s"], y=sim["Q_m3h"], mode="lines", name="Q(t)"))
    figQ.update_layout(template="plotly_white", xaxis_title="t [s]", yaxis_title="Q [m³/h]", height=340)
    st.plotly_chart(figQ, use_container_width=True)

    st.success(f"**Tiempo de reacción con hidráulica: {fmt2(t_hyd)} s**")
else:
    st.warning("Faltan parámetros para la simulación hidráulica (H0, K, ρ, r o J_eq).")


# ===========================
# 5) Exportar resumen
# ===========================
st.header("5) Exportar resumen")
summary = pd.DataFrame([{
    "TAG": str(tag),
    "J_eq_kgm2": round(J_eq, 4) if not np.isnan(J_eq) else np.nan,
    "T_nom_Nm": round(T_nom, 2),
    "r_trans": round(r_tr, 3) if not np.isnan(r_tr) else np.nan,
    "n_ini_rpm": round(nmin, 2), "n_fin_rpm": round(nmax, 2),
    "delta_n_rpm": round(delta_n, 2),
    "accel_rpm_s_torque": round(accel_torque, 2),
    "t_par_s": round(t_par, 2), "t_rampa_s": round(t_rampa, 2),
    "t_final_sin_hidraulica_s": round(t_nohyd, 2),
    "H0_m": row.get(col_H0, np.nan), "K_m_s2": row.get(col_K, np.nan),
    "rho_kgm3": row.get(col_rho, np.nan),
    "eta_a": row.get(col_eta_a, np.nan), "eta_b": row.get(col_eta_b, np.nan), "eta_c": row.get(col_eta_c, np.nan)
}])
st.dataframe(summary, use_container_width=True)
st.download_button(
    "Descargar resumen (CSV)",
    data=summary.to_csv(index=False).encode("utf-8"),
    file_name=f"resumen_{tag}.csv",
    mime="text/csv"
)


# ===========================
# 6) Conclusiones rápidas (texto + símbolos griegos correctos)
# ===========================
with st.expander("Conclusiones rápidas"):
    st.markdown(
        r"""- **Inercia equivalente** \(J_{\mathrm{eq}}\) resume los efectos del motor, transmisión y la parte reflejada del lado bomba \(\big(J_{\mathrm{driven}}+J_{\mathrm{imp}}\big)/r^2\).
- La **capacidad de aceleración por par** es \(\dot n_{\mathrm{torque}}=\frac{60}{2\pi}\frac{T_{\mathrm{nom}}}{J_{\mathrm{eq}}}\) (rpm/s).
- El **tiempo sin hidráulica** está acotado por \(t_{\mathrm{final,sin}}=\max(t_{\mathrm{par}},t_{\mathrm{rampa}})\).
- Con hidráulica, el par resistente se calcula con \(\rho\), \(g\), \(H(Q)=H_0+KQ^2\) y la eficiencia \(\eta(Q)=a+bQ+cQ^2\), y se integra en el tiempo respetando la rampa del VDF.
"""
    )
