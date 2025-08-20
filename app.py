# app.py
# ──────────────────────────────────────────────────────────────────────────────
# Memoria de Cálculo – Tiempo de reacción de bombas (VDF)
# App “todo-en-uno” con:
#  • Lectura del dataset `bombas_dataset_with_torque_params.xlsx` (hoja 0)
#  • Resumen de parámetros por TAG (motor, transmisión, bomba, sistema)
#  • 3) Respuesta inercial (sin hidráulica): fórmulas LaTeX + gráficos
#  • 4) Respuesta con hidráulica (modelo sencillo): integración J·ω̇ = Tdisp − Tload
#  • Exportación de tabla de resultados para todos los TAG
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Memoria de Cálculo – Tiempo de reacción (VDF)",
    layout="wide",
    page_icon="⚙️",
)

# =============  ESTILO  =======================================================
st.markdown("""
<style>
:root{
  --ok:#16a34a;      /* verde */
  --warn:#d97706;    /* ámbar */
  --info:#2563eb;    /* azul */
  --muted:#475569;   /* gris */
  --panel:#0b1220;   /* para hlines */
}
.badge{display:inline-flex;align-items:center;gap:.5rem;
  padding:.28rem .55rem;border-radius:999px;
  font-weight:700;font-size:.90rem;color:white;margin:.15rem .15rem}
.badge.ok{background:var(--ok);} 
.badge.warn{background:var(--warn);} 
.badge.info{background:var(--info);} 
.badge.muted{background:var(--muted);} 

.kpi{display:flex;flex-direction:column;align-items:center;justify-content:center;
  min-width:170px;padding:.6rem .9rem;margin:.25rem;border-radius:.8rem;
  background:rgba(37,99,235,.06);border:1px solid rgba(148,163,184,.25)}
.kpi .v{font-size:1.35rem;font-variant-numeric:tabular-nums; font-weight:800}
.kpi .l{font-size:.78rem;color:#9aa4b2;letter-spacing:.01em}

.hl{padding:.55rem .75rem;border-radius:.6rem;border:1px dashed rgba(148,163,184,.35);
  background:rgba(22,163,74,.08);display:inline-block}
.sep{height:8px}

.small{font-size:.9rem;color:#94a3b8}

table { font-variant-numeric: tabular-nums; }
</style>
""", unsafe_allow_html=True)


# =============  CARGA DE DATOS  ==============================================
@st.cache_data(show_spinner=False)
def load_dataset(path: str = "bombas_dataset_with_torque_params.xlsx") -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_col(df: pd.DataFrame, *hints: str, default: str | None = None) -> str | None:
    hints_low = [h.lower() for h in hints]
    for c in df.columns:
        lc = c.lower()
        if all(h in lc for h in hints_low):
            return c
    # si no encontró todas, prueba “cualquiera de”
    for c in df.columns:
        lc = c.lower()
        if any(h in lc for h in hints_low):
            return c
    return default


def value(row, name_options, default=np.nan):
    for n in name_options:
        if n in row.index:
            return row[n]
    return default


df = load_dataset()

# Mapeo de nombres (intenta ser tolerante con encabezados)
COL_TAG        = find_col(df, "tag") or df.columns[0]                 # 1ra col si no hay "TAG"
COL_R          = find_col(df, "relac", "ratio", "r")                  # relación transmisión (motor/pump)
COL_TNOM       = find_col(df, "t_nom", "torque", "nm")
COL_NMIN_M     = find_col(df, "motor", "min", "rpm")
COL_NMAX_M     = find_col(df, "motor", "max", "rpm")
COL_JM         = find_col(df, "j_m", "motor", "kgm")
COL_JDRV       = find_col(df, "j_driver", "driver", "kgm")
COL_JDRN       = find_col(df, "j_driven", "driven", "kgm")
COL_JIMP       = find_col(df, "j_imp", "impuls", "kgm")
# Hidráulica
C_H0           = find_col(df, "h0")
C_K            = find_col(df, "k", "m_s2")
C_ETA_A        = find_col(df, "eta_a")
C_ETA_B        = find_col(df, "eta_b")
C_ETA_C        = find_col(df, "eta_c")
C_RHO          = find_col(df, "rho", "kgm3")
C_NREF         = find_col(df, "n_ref")
C_QMIN         = find_col(df, "q_min")
C_QMAX         = find_col(df, "q_max")

# =============  UI – SELECCIÓN  ==============================================
st.title("Memoria de Cálculo – Tiempo de reacción de bombas (VDF)")

st.sidebar.header("Selección")
tags = df[COL_TAG].astype(str).tolist()
tag = st.sidebar.selectbox("TAG", tags, index=0)

row = df[df[COL_TAG].astype(str) == tag].iloc[0]

# Lee valores (SI)
r         = float(value(row, [COL_R], 1.0))                           # relación n_motor / n_bomba
T_nom     = float(value(row, [COL_TNOM], 0.0))
n_min_m   = float(value(row, [COL_NMIN_M], 0.0))
n_max_m   = float(value(row, [COL_NMAX_M], 0.0))
J_m       = float(value(row, [COL_JM], 0.0))
J_driver  = float(value(row, [COL_JDRV], 0.0))
J_driven  = float(value(row, [COL_JDRN], 0.0))
J_imp     = float(value(row, [COL_JIMP], 0.0))
# hidraulica
H0        = float(value(row, [C_H0], 0.0))
Ksys      = float(value(row, [C_K], 0.0))
eta_a     = float(value(row, [C_ETA_A], 0.0))
eta_b     = float(value(row, [C_ETA_B], 0.0))
eta_c     = float(value(row, [C_ETA_C], 0.0))
rho       = float(value(row, [C_RHO], 1000.0))
n_ref     = float(value(row, [C_NREF], max(n_min_m, (n_min_m+n_max_m)/2)))
Q_min     = float(value(row, [C_QMIN], 0.0))
Q_max     = float(value(row, [C_QMAX], 0.0))

# velocidades bomba desde r
n_min_p = n_min_m / r if r else 0.0
n_max_p = n_max_m / r if r else 0.0

st.caption(f"**Dataset** leido: columnas detectadas → "
           f"TAG=`{COL_TAG}` · r=`{COL_R}` · T_nom=`{COL_TNOM}` · "
           f"n_motor_min=`{COL_NMIN_M}` · n_motor_max=`{COL_NMAX_M}` · "
           f"J_m=`{COL_JM}` · J_driver=`{COL_JDRV}` · J_driven=`{COL_JDRN}` · J_imp=`{COL_JIMP}` · "
           f"H0=`{C_H0}` · K=`{C_K}` · η_a/b/c=`{C_ETA_A}/{C_ETA_B}/{C_ETA_C}` · "
           f"ρ=`{C_RHO}` · n_ref=`{C_NREF}` · Q_min/max=`{C_QMIN}/{C_QMAX}`")

# =============  1) PARÁMETROS  ===============================================
st.header("1) Parámetros de entrada")

c1,c2,c3,c4 = st.columns(4)
with c1:
    st.subheader("Motor")
    st.write(f"- **T_nom [Nm]**: {T_nom:,.2f}")
    st.write(f"- **Velocidad Motor min–max [rpm]**: {n_min_m:,.2f} – {n_max_m:,.2f}")
    st.write(f"- **J_m [kg·m²]**: {J_m:,.2f}")
with c2:
    st.subheader("Transmisión")
    st.write(f"- **Relación** \( r = n_\\mathrm{{motor}}/n_\\mathrm{{bomba}} \): {r:,.2f}")
    st.write(f"- **J_driver [kg·m²]**: {J_driver:,.2f}")
    st.write(f"- **J_driven [kg·m²]**: {J_driven:,.2f}")
with c3:
    st.subheader("Bomba")
    st.write(f"- **J_imp (impulsor) [kg·m²]**: {J_imp:,.2f}")
    st.write(f"- **Velocidad Bomba min–max [rpm]**: {n_min_p:,.2f} – {n_max_p:,.2f}")
with c4:
    st.subheader("Sistema (H–Q, η)")
    st.latex(r"H(Q) = H_0 + K \left(\frac{Q}{3600}\right)^2,\quad Q\,[m^3/h]")
    st.latex(r"\eta(Q,n) = \mathrm{clip}\left(\eta_a + \eta_b \frac{Q}{Q_\mathrm{ref}} + \eta_c \left(\frac{Q}{Q_\mathrm{ref}}\right)^2,\, [\eta_{\min},\eta_{\max}]\right)")
    st.write(f"- **H0 [m]**: {H0:,.2f}")
    st.write(f"- **K [m·s⁻²]**: {Ksys:,.2f}")
    st.write(f"- **ρ [kg/m³]**: {rho:,.2f}")
    st.write(f"- **n_ref [rpm]**: {n_ref:,.2f}")
    st.write(f"- **Q rango [m³/h]**: {Q_min:,.2f} – {Q_max:,.2f}")

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# =============  2) INERCIA EQUIVALENTE  ======================================
st.header("2) Inercia equivalente al eje del motor")

# Explicación directa (sin desplegable)
st.latex(r"""
\textbf{Definición:}\quad 
J_\mathrm{eq} \;=\; J_m \;+\; J_\mathrm{driver} \;+\; \frac{J_\mathrm{driven}+J_\mathrm{imp}}{r^2}
""")
st.caption("Las inercias del lado bomba giran a ωₚ = ωₘ/r. Igualando energías cinéticas "
           "a una ωₘ común se obtiene la división por r² del término del lado bomba.")

J_eq = J_m + J_driver + (J_driven + J_imp) / (r**2 if r else np.inf)

c21,c22 = st.columns([2,1])
with c21:
    st.latex(r"""
\textbf{Sustitución numérica:}\quad
J_\mathrm{eq} \;=\; J_m + J_\mathrm{driver} + \frac{J_\mathrm{driven}+J_\mathrm{imp}}{r^2}
\;=\; %.2f + %.2f + \frac{%.2f + %.2f}{(%.2f)^2}
""" % (J_m, J_driver, J_driven, J_imp, r))
with c22:
    st.markdown(f"<div class='kpi'><div class='l'>J_eq (kg·m²)</div>"
                f"<div class='v'>{J_eq:,.2f}</div></div>", unsafe_allow_html=True)

# Waterfall de aportes
wf = go.Figure(go.Waterfall(
    name="J_eq",
    orientation="v",
    measure=["absolute","relative","relative"],
    x=["J_m","J_driver","(J_driven+J_imp)/r²"],
    y=[J_m, J_driver, (J_driven+J_imp)/(r**2 if r else np.inf)],
    connector={"line":{"color":"#94a3b8"}}))
wf.add_hline(y=J_eq, line_dash="dot", line_color="#16a34a",
             annotation_text=f"J_eq={J_eq:,.2f} kg·m²")
wf.update_layout(height=280, yaxis_title="kg·m²", margin=dict(l=30,r=10,t=30,b=10))
st.plotly_chart(wf, use_container_width=True)

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# =============  3) RESPUESTA INERCIAL (SIN EFECTOS HIDRÁULICOS)  =============
st.header("3) Respuesta inercial (sin efectos hidráulicos)")

# Entradas compactas
cc1, cc2, cc3, cc4 = st.columns(4)
with cc1:
    n_i = st.number_input("Velocidad Motor inicial [rpm]", value=float(n_min_m), step=1.0, format="%.2f")
with cc2:
    n_f = st.number_input("Velocidad Motor final [rpm]", value=float(min(n_max_m, n_min_m + 737)), step=1.0, format="%.2f")
with cc3:
    T_disp = st.number_input("Par disponible (T_disp) [Nm]", value=float(T_nom), step=1.0, format="%.2f")
with cc4:
    ramp_vdf = st.number_input("Rampa VDF (motor) [rpm/s]", value=300.0, min_value=1.0, step=10.0, format="%.2f")

# Fórmulas
st.markdown("**Definiciones:**")
st.latex(r"\dot n_{\mathrm{torque}} = \frac{60}{2\pi}\frac{T_{\mathrm{disp}}}{J_{\mathrm{eq}}},\quad "
         r"t_{\mathrm{par}} = \frac{\Delta n}{\dot n_{\mathrm{torque}}},\quad "
         r"t_{\mathrm{rampa}} = \frac{\Delta n}{\mathrm{rampa}_{\mathrm{VDF}}},\quad "
         r"t_{\mathrm{final}} = \max(t_{\mathrm{par}}, t_{\mathrm{rampa}})")
delta_n = abs(n_f - n_i)
n_dot_torque = (60.0/(2*np.pi))*(T_disp/max(1e-12, J_eq))  # rpm/s
t_par   = delta_n/max(1e-12, n_dot_torque)
t_ramp  = delta_n/max(1e-12, ramp_vdf)
t_final = max(t_par, t_ramp)

# Chips con valores (mismos símbolos)
st.markdown(
    f"<span class='badge info'>Δn: {delta_n:,.2f} rpm</span>"
    f"<span class='badge ok'>ṅ_torque: {n_dot_torque:,.2f} rpm/s</span>"
    f"<span class='badge ok'>t_par: {t_par:,.2f} s</span>"
    f"<span class='badge ok'>t_rampa: {t_ramp:,.2f} s</span>"
    f"<span class='badge warn'>t_final: {t_final:,.2f} s</span>",
    unsafe_allow_html=True
)

# Perfil n(t)
def speed_profile(n_i, n_f, n_dot_eff):
    dn = n_f - n_i
    t_end = abs(dn) / max(1e-12, n_dot_eff)
    t = np.linspace(0, t_end, 200)
    n = n_i + np.sign(dn)*n_dot_eff*t
    return t, n, t_end

n_dot_eff = min(n_dot_torque, ramp_vdf)
t_prof, n_prof, t_end = speed_profile(n_i, n_f, n_dot_eff)

fig = go.Figure()
fig.add_trace(go.Scatter(x=t_prof, y=n_prof, mode="lines", name="n(t)"))
fig.add_vline(x=t_par, line_dash="dot", line_color="#d97706", annotation_text="t_par")
fig.add_vline(x=t_ramp, line_dash="dot", line_color="#2563eb", annotation_text="t_rampa")
fig.add_vline(x=t_final, line_width=3, line_color="#16a34a",
              annotation_text=f"t_final={t_final:.2f}s")
fig.update_layout(height=340, xaxis_title="t [s]", yaxis_title="n_motor [rpm]",
                  margin=dict(l=30,r=10,t=10,b=10))
st.plotly_chart(fig, use_container_width=True)

# Comparación de tiempos
bar = go.Figure(go.Bar(x=["t_par","t_rampa"], y=[t_par, t_ramp],
                       marker_color=["#d97706","#2563eb"]))
bar.add_hline(y=t_final, line_dash="dot", line_color="#16a34a",
              annotation_text=f"t_final {t_final:.2f}s")
bar.update_layout(height=280, yaxis_title="s", margin=dict(l=30,r=10,t=10,b=10))
st.plotly_chart(bar, use_container_width=True)

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# =============  4) RESPUESTA CON HIDRÁULICA (MODELO SIMPLE)  =================
st.header("4) Respuesta con hidráulica (modelo simple)")

# Modelo de sistema y eficiencia
def H_sys(Q_m3h):
    return H0 + Ksys * (Q_m3h/3600.0)**2

Q_ref = (Q_min + Q_max)/2 if Q_max > Q_min else max(Q_min, Q_max, 1.0)

def eta_of(Q_m3h, n_rpm, eta_min=0.30, eta_max=0.90, eta_beta=0.0):
    # polinomio en Q/Q_ref (independiente de n en esta versión)
    x = (Q_m3h / max(1e-9, Q_ref))
    eta = eta_a + eta_b*x + eta_c*(x**2) + eta_beta
    return float(np.clip(eta, eta_min, eta_max))

# Aproximación por afinidad para Q(n): Q ≈ α·n
alpha_Qn = Q_ref / max(1e-9, n_ref)   # m3/h por rpm
def Q_from_n(n_rpm):  # dentro del rango
    return float(np.clip(alpha_Qn * n_rpm, Q_min, Q_max))

# Dinámica J·ω̇ = Tdisp − Tload(ω, Q)
def simulate_with_hydraulics(n_i_rpm, n_f_rpm, r, T_disp, J_eq,
                             ramp_motor_rpms=300.0, dt=0.01, tmax=30.0):
    t = 0.0
    n_m = n_i_rpm
    omega_m = n_m * 2*np.pi/60
    history = {"t":[], "n_motor":[], "n_pump":[], "Q":[], "H":[], "eta":[], "T_load":[], "T_disp":[]}

    while t < tmax:
        n_set = n_i_rpm + np.sign(n_f_rpm-n_i_rpm)*ramp_motor_rpms*t
        # tope por orden de referencia
        if (n_f_rpm >= n_i_rpm and n_m >= n_f_rpm) or (n_f_rpm < n_i_rpm and n_m <= n_f_rpm):
            break

        # n_pump por relación
        n_p = n_m / max(1e-9, r)
        Q   = Q_from_n(n_p)  # screening simple por afinidad
        H   = H_sys(Q)
        eta = eta_of(Q, n_p)

        # Par resistente hidráulico ≈ (ρ g Q H) / (η · ω_pump) reflejado al motor:
        omega_p = max(1e-9, n_p*2*np.pi/60)
        P_h = rho*9.81*(Q/3600.0)*H              # W (kg/m^3 * m/s * m) = N·m/s
        T_load_p = P_h / max(1e-9, eta*omega_p)  # N·m en eje bomba
        # reflejado al motor por r: T_load_m = T_load_p  (par no cambia)  → válido para r por poleas
        T_load_m = T_load_p

        domega_m = (T_disp - T_load_m) / max(1e-9, J_eq)
        omega_m  = omega_m + domega_m*dt

        # Limita el incremento por rampa del VDF
        n_m_next = omega_m*60/(2*np.pi)
        dn_max   = ramp_motor_rpms*dt * np.sign(n_f_rpm-n_i_rpm)
        n_m_next = n_m + np.clip(n_m_next-n_m, -abs(dn_max), abs(dn_max))

        # registro
        history["t"].append(t)
        history["n_motor"].append(n_m)
        history["n_pump"].append(n_p)
        history["Q"].append(Q)
        history["H"].append(H)
        history["eta"].append(eta)
        history["T_load"].append(T_load_m)
        history["T_disp"].append(T_disp)

        n_m = n_m_next
        t  += dt

    # tiempo total
    return pd.DataFrame(history)

st.caption("Modelo: \(J_{eq}\,\dot\omega_m = T_{disp} - T_{load}(Q,H,\eta)\) y \(Q(n)\) por afinidad "
           "(\(Q\propto n\)) con recorte al rango [Q_min, Q_max].")

c4a,c4b,c4c = st.columns(3)
with c4a:
    dt = st.number_input("Δt integración [s]", value=0.01, min_value=0.001, step=0.01, format="%.3f")
with c4b:
    tmax = st.number_input("t_max [s]", value=15.0, min_value=1.0, step=1.0, format="%.1f")
with c4c:
    run_sim = st.checkbox("Ejecutar simulación hidráulica", value=True)

if run_sim:
    hist = simulate_with_hydraulics(n_i, n_f, r, T_disp, J_eq, ramp_motor_rpms=ramp_vdf, dt=dt, tmax=tmax)

    # Gráfico n_motor y Q vs tiempo
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=hist["t"], y=hist["n_motor"], name="n_motor [rpm]", yaxis="y1"))
    fig2.add_trace(go.Scatter(x=hist["t"], y=hist["Q"], name="Q [m³/h]", yaxis="y2"))
    fig2.update_layout(
        height=360,
        xaxis=dict(title="t [s]"),
        yaxis=dict(title="n_motor [rpm]", side="left"),
        yaxis2=dict(title="Q [m³/h]", overlaying="y", side="right"),
        margin=dict(l=30,r=10,t=10,b=10)
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Par disponible vs par resistente
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=hist["t"], y=hist["T_disp"], name="T_disp [Nm]"))
    fig3.add_trace(go.Scatter(x=hist["t"], y=hist["T_load"], name="T_load [Nm]"))
    fig3.update_layout(height=300, xaxis_title="t [s]", yaxis_title="Par [Nm]",
                       margin=dict(l=30,r=10,t=10,b=10))
    st.plotly_chart(fig3, use_container_width=True)

    # Eficiencia y H
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=hist["t"], y=hist["eta"], name="η [-]"))
    fig4.add_trace(go.Scatter(x=hist["t"], y=hist["H"], name="H [m]"))
    fig4.update_layout(height=260, xaxis_title="t [s]", margin=dict(l=30,r=10,t=10,b=10))
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown(
        f"<div class='kpi'><div class='l'>Tiempo de simulación hasta alcanzar n_f</div>"
        f"<div class='v'>{(hist['t'].iloc[-1] if len(hist)>0 else 0):.2f} s</div></div>",
        unsafe_allow_html=True
    )

st.markdown("<div class='sep'></div>", unsafe_allow_html=True)

# =============  5) TABLA RESUMEN Y EXPORT  ===================================
st.header("5) Tabla resumen por TAG (rampa y par actuales)")

def inertial_times_for_row(row: pd.Series, ramp_vdf: float) -> dict:
    # safe access
    def v(cols, default=0.0): 
        for c in cols:
            if c in row.index and pd.notna(row[c]): return float(row[c])
        return default

    r_   = v([COL_R], 1.0)
    Jm   = v([COL_JM], 0.0)
    Jdrv = v([COL_JDRV], 0.0)
    Jdrn = v([COL_JDRN], 0.0)
    Jimp = v([COL_JIMP], 0.0)
    nmin = v([COL_NMIN_M], 0.0)
    nmax = v([COL_NMAX_M], 0.0)
    Tn   = v([COL_TNOM], 0.0)

    Jeq  = Jm + Jdrv + (Jdrn + Jimp)/(r_**2 if r_ else np.inf)
    n_i_ = nmin
    n_f_ = nmax if nmax>nmin else nmin
    dn   = abs(n_f_-n_i_)
    n_dot_tq = (60/(2*np.pi))*Tn/max(1e-12,Jeq)
    tpar = dn/max(1e-12,n_dot_tq)
    tramp= dn/max(1e-12,ramp_vdf)
    tf  = max(tpar,tramp)
    return dict(
        TAG=row[COL_TAG],
        r=r_, J_eq=Jeq, T_nom=Tn, n_i=n_i_, n_f=n_f_, dn=dn,
        n_dot_torque=n_dot_tq, t_par=tpar, t_rampa=tramp, t_final=tf
    )

rows = [inertial_times_for_row(rw, ramp_vdf) for _,rw in df.iterrows()]
tbl = pd.DataFrame(rows)
st.dataframe(tbl.style.format({
    "r":"{:.2f}","J_eq":"{:.2f}","T_nom":"{:.2f}","n_i":"{:.2f}","n_f":"{:.2f}",
    "dn":"{:.2f}","n_dot_torque":"{:.2f}","t_par":"{:.2f}","t_rampa":"{:.2f}","t_final":"{:.2f}"
}), use_container_width=True, hide_index=True)

buf = io.StringIO()
tbl.to_csv(buf, index=False)
st.download_button("⬇️ Descargar resumen (CSV)", buf.getvalue().encode("utf-8"),
                   file_name="resumen_reaccion_por_TAG.csv", mime="text/csv")

# =============  6) NOTAS  =====================================================
with st.expander("Notas y supuestos del modelo"):
    st.markdown("""
- **Zona de par constante** del motor en el rango analizado (VDF 25–50 Hz aprox.).
- **Inercia equivalente** reflejada al eje del motor: \( J_{eq}=J_m+J_{driver}+\frac{J_{driven}+J_{imp}}{r^2} \).
- **Carga hidráulica**: \( H(Q)=H_0+K(Q/3600)^2 \), \( \eta(Q)=\eta_a+\eta_b(Q/Q_{ref})+\eta_c(Q/Q_{ref})^2 \) recortada a \([0.30,0.90]\).
- **Caudal por afinidad** (screening): \( Q \propto n \). Se usa \( Q_{ref}/n_{ref} \) para escalar y se recorta a \([Q_{min},Q_{max}]\).
- **Par resistente** en eje bomba: \( T_{load}=\frac{\rho g\,Q\,H(Q)}{\eta\,\omega_{pump}} \).
- Integración explícita con paso fijo \( \Delta t \); la rampa del VDF limita \( |\Delta n| \leq \mathrm{rampa}_{VDF}\,\Delta t \).
- Para resultados de precisión industrial se recomienda: **curva H–Q de bomba medida** o del fabricante (varias
  curvas a distintas rpm), pérdidas distribuidas/locales actualizadas y validación con datos de planta.
""")

