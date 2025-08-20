# -*- coding: utf-8 -*-
# ─────────────────────────────────────────────────────────────────────────────
# Memoria de Cálculo – Tiempo de reacción de bombas con VDF
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import io
import math
import re
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# Configuración base
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Memoria de Cálculo – Tiempo de reacción (VDF)",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATASET = "bombas_dataset_with_torque_params.xlsx"  # en la raíz del repo

# Colores/estilo liviano
PRIMARY = "#178F53"
MUTED = "#888888"

# ─────────────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────────────

def get_num(x, default=0.0) -> float:
    """Convierte lo que venga (número, string con coma/puntos, NaN) a float."""
    if pd.isna(x):
        return float(default)
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace(" ", "").replace("\u00a0", "")
    s = s.replace(",", ".")
    s = re.sub(r"[^\d\.\-eE+]", "", s)
    try:
        return float(s)
    except Exception:
        return float(default)

def badge(value: str, unit: str = "", title: str = "") -> None:
    """Chip verde con SOLO el valor (lo pedido)."""
    txt = f"{value} {unit}".strip()
    st.markdown(
        f"""
        <div style="
            display:inline-block;background:#E9F8F0;color:{PRIMARY};
            padding:6px 10px;border-radius:10px;font-weight:700;">
            {txt}
        </div>
        """,
        unsafe_allow_html=True,
    )
    if title:
        st.caption(title)

@st.cache_data(show_spinner=False)
def load_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_excel(path)  # raíz del repo
    # Asegurar columnas con la capitalización tal cual nos diste
    required = [
        "TAG", "r_trans", "motorpower_kw", "t_nom_nm", "motor_j_kgm2",
        "impeller_j_kgm2",
        "driverpulley_j_kgm2", "driverbushing_j_kgm2",
        "drivenpulley_j_Kgm2", "drivenbushing_j_Kgm2",
        "motor_n_min_rpm", "motor_n_max_rpm",
        "pump_n_min_rpm", "pump_n_max_rpm",
        "H0_m", "K_m_s2",
        "eta_a", "eta_b", "eta_c",
        "Q_min_m3h", "Q_max_m3h", "Q_ref_m3h",
        "n_ref_rpm", "rho_kgm3",
        "eta_beta", "eta_min_clip", "eta_max_clip",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Faltan columnas en el dataset: {missing}")
        st.stop()
    return df

def eta_poly(q_m3h: float, row: pd.Series, scale: Dict[str, float]) -> float:
    """
    Eficiencia 'suave' con los parámetros del dataset:
      eta = clip( a + b*(Q/Qref)^β + c*(Q/Qref)^(2β), [eta_min, eta_max] )
    Se ajusta por ±30% con multiplicadores en 'scale'.
    """
    Qref = max(1e-6, get_num(row["Q_ref_m3h"]))
    beta = get_num(row["eta_beta"], 1.0)
    a = get_num(row["eta_a"]) * scale["eta_a"]
    b = get_num(row["eta_b"]) * scale["eta_b"]
    c = get_num(row["eta_c"]) * scale["eta_c"]

    x = max(0.0, q_m3h / Qref)
    eta = a + b * (x ** beta) + c * (x ** (2.0 * beta))

    # clip por dataset y por rango físico
    emin = get_num(row["eta_min_clip"], 0.05)
    emax = get_num(row["eta_max_clip"], 0.92)
    return float(np.clip(eta, emin, emax))

def head_curve(q_m3h: float, row: pd.Series, scale: Dict[str, float]) -> float:
    """
    H(Q) = H0 + K * (Q/3600)^2
    con ajustes relativos de H0 y K.
    """
    H0 = get_num(row["H0_m"]) * scale["H0_m"]
    K = get_num(row["K_m_s2"]) * scale["K_m_s2"]
    q_m3s = max(0.0, q_m3h) / 3600.0
    return H0 + K * (q_m3s ** 2)

def affinity_Q(n_pump_rpm: float, row: pd.Series) -> float:
    """
    Q(n) por afinidad: Q = Qref * (n/n_ref), con recorte [Qmin, Qmax].
    """
    Qref = get_num(row["Q_ref_m3h"])
    nref = max(1.0, get_num(row["n_ref_rpm"]))
    q = Qref * (n_pump_rpm / nref)
    qmin = get_num(row["Q_min_m3h"])
    qmax = get_num(row["Q_max_m3h"])
    return float(np.clip(q, qmin, qmax))

def motor_torque_available(n_motor_rpm: float, row: pd.Series) -> float:
    """
    Curva de par disponible:
      — Tomamos t_nom_nm como par constante (caso típico) y
      — limitamos por potencia: T <= P/ω.
    """
    T_nom = get_num(row["t_nom_nm"])
    P_w = 1000.0 * get_num(row["motorpower_kw"])
    omega = max(1e-6, (2.0 * math.pi / 60.0) * n_motor_rpm)
    T_power = P_w / omega
    return float(min(T_nom, T_power))

def J_equiv(row: pd.Series) -> float:
    """
    J_eq al eje motor:
      J_m + J_driver + (J_driven + J_imp) / r^2
    """
    r = max(1e-6, get_num(row["r_trans"]))
    Jm = get_num(row["motor_j_kgm2"])
    Jdriver = get_num(row["driverpulley_j_kgm2"]) + get_num(row["driverbushing_j_kgm2"])
    Jdriven = get_num(row["drivenpulley_j_Kgm2"]) + get_num(row["drivenbushing_j_Kgm2"])
    Jimp = get_num(row["impeller_j_kgm2"])
    return float(Jm + Jdriver + (Jdriven + Jimp) / (r ** 2))

def integrate_response(
    row: pd.Series,
    rpm_pump_ini: float,
    rpm_pump_fin: float,
    rampa_rpmps: float,
    scale: Dict[str, float],
    dt: float = 0.02,
    max_time: float = 120.0,
) -> Dict[str, np.ndarray | float | bool]:
    """
    Integra la dinámica con limitación conjunta por par y por rampa VDF.
    Devuelve series de t, n_pump, Q, P_hid y diagnósticos.
    """
    r = max(1e-6, get_num(row["r_trans"]))
    rho = get_num(row["rho_kgm3"]) * scale["rho"]
    g = 9.81

    # Estados
    n_m = rpm_pump_ini * r  # rpm motor
    n_p = rpm_pump_ini
    omega_m = (2.0 * math.pi / 60.0) * n_m
    J = J_equiv(row)

    # Aceleración máxima por rampa (rad/s^2)
    a_ramp = (2.0 * math.pi / 60.0) * max(0.0, rampa_rpmps)

    t_list, n_p_list, Q_list, P_list = [0.0], [n_p], [affinity_Q(n_p, row)], [0.0]

    t = 0.0
    target = rpm_pump_fin
    stall_counter = 0
    last_n_p = n_p

    while (n_p < target - 1e-3) and (t < max_time):
        # Hidráulica a la velocidad actual
        Q = affinity_Q(n_p, row)
        H = head_curve(Q, row, scale)
        eta = max(1e-3, eta_poly(Q, row, scale))
        omega_p = max(1e-6, (2.0 * math.pi / 60.0) * n_p)
        # Potencia hidráulica y par bomba
        P_hid = rho * g * (Q / 3600.0) * H / max(1e-6, eta)  # W
        T_pump = P_hid / omega_p  # N·m
        # Par visto por el motor
        T_load_m = T_pump / r
        T_disp = motor_torque_available(n_m, row)
        # Aceleración por par disponible
        a_torque = (T_disp - T_load_m) / max(1e-9, J)  # rad/s^2

        # Limitación conjunta: rampa y par
        a_eff = min(a_ramp, a_torque) if a_torque > 0 else a_torque

        # Integración explícita
        omega_m = max(0.0, omega_m + a_eff * dt)
        n_m = omega_m * 60.0 / (2.0 * math.pi)
        n_p = n_m / r

        t += dt
        t_list.append(t)
        n_p_list.append(n_p)
        Q_list.append(Q)
        P_list.append(P_hid)

        # Detección de “no converge” (sin avance apreciable)
        if abs(n_p - last_n_p) < 1e-4:
            stall_counter += 1
        else:
            stall_counter = 0
        last_n_p = n_p
        if stall_counter * dt > 1.5:  # 1.5 s sin avanzar ~ “se plantó”
            break

    result = {
        "t": np.array(t_list),
        "n_p": np.array(n_p_list),
        "Q": np.array(Q_list),
        "P": np.array(P_list),
        "t_total": t_list[-1],
        "reached": (n_p_list[-1] >= target - 1e-3),
        "t_rampa": (rpm_pump_fin - rpm_pump_ini) / max(1e-6, rampa_rpmps),
    }
    return result

def csv_bytes_from_df(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

# ─────────────────────────────────────────────────────────────────────────────
# UI – Sidebar: selección de TAG
# ─────────────────────────────────────────────────────────────────────────────
df = load_dataset(DATASET)
tags = df["TAG"].astype(str).tolist()
tag = st.sidebar.selectbox("Selecciona TAG", tags, index=0, key="sel_tag")

row = df[df["TAG"].astype(str) == str(tag)].iloc[0]

# ─────────────────────────────────────────────────────────────────────────────
# Sección 1 — Motor, Transmisión y Bomba
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## 1) Motor, transmisión y bomba")

c1, c2, c3, c4 = st.columns(4)
with c1:
    badge(f"{get_num(row['motorpower_kw']):.2f}", "kW")
    st.caption("Potencia del motor")
with c2:
    badge(f"{get_num(row['t_nom_nm']):.1f}", "N·m")
    st.caption("Par nominal (límite base)")
with c3:
    badge(f"{get_num(row['r_trans']):.3f}", "")
    st.caption("Relación r = n_motor/n_bomba")
with c4:
    badge(f"{J_equiv(row):.4f}", "kg·m²")
    st.caption("Inercia equivalente al eje motor")

c5, c6, c7, c8 = st.columns(4)
with c5:
    badge(f"{get_num(row['motor_n_min_rpm']):.0f}", "rpm")
    st.caption("n_motor min")
with c6:
    badge(f"{get_num(row['motor_n_max_rpm']):.0f}", "rpm")
    st.caption("n_motor máx")
with c7:
    badge(f"{get_num(row['pump_n_min_rpm']):.0f}", "rpm")
    st.caption("n_bomba min")
with c8:
    badge(f"{get_num(row['pump_n_max_rpm']):.0f}", "rpm")
    st.caption("n_bomba máx @50 Hz (dataset)")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Sección 3 — Rampa del VDF (se mantiene en 4)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## 3) Rampa del VDF")

rampa_vdf = st.slider(
    "Pendiente de rampa (motor) [rpm/s]",
    min_value=10.0,
    max_value=1000.0,
    value=200.0,
    step=5.0,
    help="Se usará también en la sección 4 al integrar con hidráulica.",
    key="rampa_vdf",
)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Sección 4 — Sistema (H–Q, η) y densidad – ajustes e integración
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## 4) Sistema (H–Q, η) y densidad — ajustes y respuesta con hidráulica")

st.caption(
    "Ajustes relativos (±30%) con paso de 1%. Los valores efectivos afectan la integración."
)

def pct_control(label: str, base_value: float, key: str) -> float:
    """Control con input + botones ±1% (limitado ±30%). Devuelve multiplicador."""
    col = st.columns([1, 1, 1, 2])
    with col[0]:
        minus = st.button("–", key=f"minus_{key}", help="−1%")
    with col[1]:
        plus = st.button("+", key=f"plus_{key}", help="+1%")
    with col[2]:
        val = st.number_input(
            f"{label} (ajuste %)",
            min_value=-30,
            max_value=30,
            value=0,
            step=1,
            key=f"pct_{key}",
            label_visibility="visible",
        )
    # actualizar por botones
    if minus:
        st.session_state[f"pct_{key}"] = max(-30, val - 1)
    if plus:
        st.session_state[f"pct_{key}"] = min(30, val + 1)
    pct = st.session_state[f"pct_{key}"]
    mult = 1.0 + pct / 100.0
    with col[3]:
        # chip con el valor efectivo
        if key == "H0":
            badge(f"{base_value*mult:.2f}", "m")
        elif key == "K":
            badge(f"{base_value*mult:.2f}", "m·s²/m⁶")
        elif key == "rho":
            badge(f"{base_value*mult:.0f}", "kg/m³")
        else:
            badge(f"{mult*100:.0f}", "%")
        st.caption(f"ajuste: {mult*100:.0f}%")
    return mult

cA, cB, cC = st.columns(3)
with cA:
    scale_H0 = pct_control("H0 [m] (±30%)", get_num(row["H0_m"]), "H0")
with cB:
    scale_K = pct_control("K [m·s²/m⁶] (±30%)", get_num(row["K_m_s2"]), "K")
with cC:
    scale_rho = pct_control("ρ pulpa [kg/m³] (±30%)", get_num(row["rho_kgm3"]), "rho")

cE1, cE2, cE3 = st.columns(3)
with cE1:
    scale_eta_a = pct_control("η_a (±30%)", get_num(row["eta_a"]), "eta_a")
with cE2:
    scale_eta_b = pct_control("η_b (±30%)", get_num(row["eta_b"]), "eta_b")
with cE3:
    scale_eta_c = pct_control("η_c (±30%)", get_num(row["eta_c"]), "eta_c")

scale = {
    "H0_m": scale_H0,
    "K_m_s2": scale_K,
    "eta_a": scale_eta_a,
    "eta_b": scale_eta_b,
    "eta_c": scale_eta_c,
    "rho": scale_rho,
}

st.markdown("#### Rango de BOMBA a evaluar [rpm] (inicio → fin)")
n_p_min = 0.0
n_p_max_50 = get_num(row["pump_n_max_rpm"])
rng = st.slider(
    "Selecciona rango",
    min_value=float(n_p_min),
    max_value=float(n_p_max_50),
    value=(float(n_p_min), float(n_p_max_50)),
    step=1.0,
    label_visibility="collapsed",
    key="rng_pump",
)

n_ini, n_fin = rng
st.caption(f"Rampa de VDF (motor) usada (de 3): **{rampa_vdf:.0f} rpm/s**")

# Integración
res = integrate_response(
    row=row,
    rpm_pump_ini=n_ini,
    rpm_pump_fin=n_fin,
    rampa_rpmps=rampa_vdf,
    scale=scale,
    dt=0.02,
    max_time=600.0,
)

# Diagnóstico y resultados clave
cR1, cR2, cR3 = st.columns(3)
with cR1:
    badge(f"{(n_fin - n_ini):.0f}", "rpm")
    st.caption("Δn evaluado")
with cR2:
    badge(f"{res['t_rampa']:.2f}", "s")
    st.caption("t_por_rampa (solo VDF)")
with cR3:
    badge(f"{res['t_total']:.2f}", "s")
    st.caption("t_por_par (con hidráulica)")

if not res["reached"]:
    st.error(
        "El cálculo hidráulico **no converge** en todo el rango (par disponible insuficiente). "
        "Reduce el rango final o aumenta la rampa solo si no te limita el par."
    )

# Limitante
limitante = "rampa" if res["t_rampa"] >= res["t_total"] else "par (hidráulica)"
st.info(f"**Limitante** en este seteo: **{limitante}**.")

# Gráfico (Q, rpm, Potencia)
fig = go.Figure()

t = res["t"]
Q = res["Q"]
n_p = res["n_p"]
P = res["P"] / 1000.0  # kW

fig.add_trace(go.Scatter(x=t, y=Q, name="Q [m³/h]", line=dict(width=3)))
fig.update_yaxes(title_text="Q [m³/h]", showgrid=True, zeroline=False)

fig.add_trace(go.Scatter(x=t, y=n_p, name="n bomba [rpm]", yaxis="y2", line=dict(width=2, dash="dot")))
fig.add_trace(go.Scatter(x=t, y=P, name="P hidráulica [kW]", yaxis="y3", line=dict(width=2, dash="dash")))

fig.update_layout(
    template="plotly_white",
    margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    xaxis=dict(title="Tiempo [s]"),
    yaxis=dict(title="Q [m³/h]"),
    yaxis2=dict(title="n bomba [rpm]", overlaying="y", side="right"),
    yaxis3=dict(title="P [kW]", overlaying="y", side="right", anchor="free", position=1.10),
    width=None,
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Reporte para TODOS los TAG (con rampa y ajustes actuales)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Reporte – todos los TAG con el seteo actual")

def compute_row_report(row: pd.Series) -> Dict[str, float]:
    n_ini_loc, n_fin_loc = n_ini, n_fin
    res_loc = integrate_response(
        row=row,
        rpm_pump_ini=n_ini_loc,
        rpm_pump_fin=n_fin_loc,
        rampa_rpmps=rampa_vdf,
        scale=scale,
        dt=0.02,
        max_time=600.0,
    )
    return {
        "TAG": row["TAG"],
        "r": get_num(row["r_trans"]),
        "J_eq_kgm2": J_equiv(row),
        "n_ini_rpm": n_ini_loc,
        "n_fin_rpm": n_fin_loc,
        "t_por_rampa_s": res_loc["t_rampa"],
        "t_por_par_s": res_loc["t_total"],
        "limitante": "rampa" if res_loc["t_rampa"] >= res_loc["t_total"] else "par",
        "alcanzado": bool(res_loc["reached"]),
    }

rep_rows: List[Dict[str, float]] = []
with st.spinner("Calculando…"):
    for _, r_ in df.iterrows():
        rep_rows.append(compute_row_report(r_))

rep_df = pd.DataFrame(rep_rows)
st.dataframe(rep_df, use_container_width=True, hide_index=True)

# Para evitar el error “Missing file … .csv” usamos bytes en memoria
csv_all = csv_bytes_from_df(rep_df)
st.download_button(
    label="⬇️ Descargar reporte (todos los TAG, con rampa y ajustes actuales)",
    data=csv_all,
    file_name=f"reporte_TAGs_rampa_{int(rampa_vdf)}rpmps.csv",
    mime="text/csv",
    key="download_all_tags",
)
