import os
import unicodedata
import re
import numpy as np
import pandas as pd
import streamlit as st

# --------- (opcional) soporte de gráficos sin romper si falta plotly ----------
try:
    import plotly.graph_objects as go
    HAVE_PLOTLY = True
except Exception:
    go = None
    HAVE_PLOTLY = False
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Memoria de Cálculo – Bombas (VDF)", layout="wide")

DATA_FILE = "bombas_dataset_with_torque_params.xlsx"
SHEET = "dataSet"  # según indicaste, sólo hay esta tabla


# -------------------------- utilidades de parsing -----------------------------
def slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s.strip().lower())
    return re.sub(r"_+", "_", s).strip("_")


def to_float(x):
    """
    Convierte a float aceptando:
    - cadenas con coma decimal '1.234,56'
    - NaN / None
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if s == "":
        return np.nan
    # quita separadores de miles y usa punto como decimal
    s = s.replace(" ", "")
    # si hay coma y punto, asume punto miles, coma decimal
    if "," in s and "." in s:
        # remove thousands dots, replace decimal comma
        parts = s.split(",")
        s = "".join(parts[:-1]).replace(".", "") + "." + parts[-1]
    else:
        # sólo coma -> decimal
        if "," in s and "." not in s:
            s = s.replace(".", "").replace(",", ".")
        # sólo punto -> ya está OK
    try:
        return float(s)
    except Exception:
        return np.nan


def pick_col(df_cols, candidates, deny_tokens=None):
    """
    Devuelve el primer nombre de columna (normalizado) que exista en df_cols
    y que NO contenga ningún deny_token (por ejemplo 'kw' para evitar Power).
    """
    deny_tokens = deny_tokens or []
    for c in candidates:
        if c in df_cols:
            if any(tok in c for tok in deny_tokens):
                continue
            return c
    return None


def sum_cols(row, names):
    vals = [to_float(row.get(c)) for c in names if c in row.index]
    vals = [v for v in vals if not np.isnan(v)]
    return np.sum(vals) if vals else np.nan
# -----------------------------------------------------------------------------


# -------------------------------- carga dataset ------------------------------
if not os.path.exists(DATA_FILE):
    st.error(f"No encuentro **{DATA_FILE}** en la raíz de la app.")
    st.stop()

# lee
try:
    df_raw = pd.read_excel(DATA_FILE, sheet_name=SHEET)
except Exception:
    # si no existe la hoja 'dataSet', usa la primera
    df_raw = pd.read_excel(DATA_FILE)

# normaliza encabezados
df = df_raw.copy()
df.columns = [slug(c) for c in df.columns]

# convierte a float todo lo que parezca numérico (de forma tolerante)
for c in df.columns:
    df[c] = df[c].apply(to_float if df[c].dtype == object else lambda x: x)

# ------------------------ detectar columnas por nombre -----------------------
cols = set(df.columns)

# TAG
col_tag = pick_col(cols, ["tag", "eq_no", "eqno", "eq_n_o", "eq_nro", "eqno_", "eq"])
if col_tag is None:
    # fallback: primera columna
    col_tag = df.columns[0]

# relación de transmisión (evitar confundir con potencia)
# nombres posibles que vimos: 'relaciontransmision', 'r', 'ratio'
col_r = pick_col(
    cols,
    ["relaciontransmision", "relacion_transmision", "rel_transmision", "r", "ratio", "relacion_r"],
    deny_tokens=["kw", "power", "potencia"],
)

# inercia motor
col_jm = pick_col(
    cols,
    ["motor_j_kgm2", "j_m_kgm2", "jm_kgm2", "j_m", "j_motor", "j_motor_kgm2"],
    deny_tokens=["kw", "power", "potencia"],
)

# inercia polea + bushing lado motor (driver)
col_jdriver_total = pick_col(
    cols, ["j_driver_total_kgm2", "j_driver_total", "jdriver_total_kgm2"]
)
col_jdriver_pul = pick_col(cols, ["j_polea_conductora_kgm2", "j_driver_pulley_kgm2"])
col_jdriver_bush = pick_col(cols, ["j_manguito_motor_kgm2", "j_driver_bushing_kgm2"])

# inercia polea + bushing lado bomba (driven)
col_jdriven_total = pick_col(
    cols, ["j_driven_total_kgm2", "j_driven_total", "jdriven_total_kgm2"]
)
col_jdriven_pul = pick_col(cols, ["j_polea_bomba_kgm2", "j_driven_pulley_kgm2"])
col_jdriven_bush = pick_col(cols, ["j_manguito_bomba_kgm2", "j_driven_bushing_kgm2"])

# inercia impulsor
col_jimp = pick_col(cols, ["impeller_j_kgm2", "j_imp", "j_imp_kgm2", "jimp_kgm2"])

# algunas columnas informativas (no usadas en J_eq)
col_kw = pick_col(cols, ["motorpower_kw", "kw", "power_kw", "potencia_kw"])

# Validaciones rápidas
problems = []
if col_r is None:
    problems.append("No se encontró columna de **relación de transmisión** (por ejemplo `relaciontransmision` o `r`).")
if col_jm is None:
    problems.append("No se encontró columna de **J_m (kg·m²)** (por ejemplo `motor_j_kgm2`).")
if col_jimp is None:
    problems.append("No se encontró columna de **J_imp (kg·m²)** (por ejemplo `impeller_j_kgm2`).")

if problems:
    st.error("Revise el dataset:\n\n- " + "\n- ".join(problems))
    st.stop()

# --------------- arma columnas consolidadas para el cálculo ------------------
def compute_row_values(row):
    r = to_float(row[col_r]) if col_r else np.nan

    # J_m
    Jm = to_float(row[col_jm]) if col_jm else np.nan

    # J_driver (total o suma de parciales)
    if col_jdriver_total and not np.isnan(to_float(row[col_jdriver_total])):
        Jdriver = to_float(row[col_jdriver_total])
    else:
        Jdriver = sum_cols(row, [col_jdriver_pul, col_jdriver_bush])

    # J_driven (total o suma de parciales)
    if col_jdriven_total and not np.isnan(to_float(row[col_jdriven_total])):
        Jdriven = to_float(row[col_jdriven_total])
    else:
        Jdriven = sum_cols(row, [col_jdriven_pul, col_jdriven_bush])

    # J_imp
    Jimp = to_float(row[col_jimp]) if col_jimp else np.nan

    return r, Jm, Jdriver, Jdriven, Jimp


df["_r"] = np.nan
df["_jm"] = np.nan
df["_jdriver"] = np.nan
df["_jdriven"] = np.nan
df["_jimp"] = np.nan

for i, row in df.iterrows():
    r, Jm, Jdriver, Jdriven, Jimp = compute_row_values(row)
    df.at[i, "_r"] = r
    df.at[i, "_jm"] = Jm
    df.at[i, "_jdriver"] = Jdriver
    df.at[i, "_jdriven"] = Jdriven
    df.at[i, "_jimp"] = Jimp

# J_eq
df["_jeq"] = df["_jm"] + df["_jdriver"] + (df["_jdriven"] + df["_jimp"]) / (df["_r"] ** 2)

# ------------------------------ UI ------------------------------------------
st.title("Memoria de Cálculo – Tiempo de reacción de bombas (VDF)")

# selector por TAG
tags = df[col_tag].astype(str).tolist()
tag_sel = st.sidebar.selectbox("Selecciona TAG", tags, index=0)

row = df[df[col_tag].astype(str) == str(tag_sel)].iloc[0]

r = float(row["_r"])
Jm = float(row["_jm"])
Jdriver = float(row["_jdriver"]) if not np.isnan(row["_jdriver"]) else 0.0
Jdriven = float(row["_jdriven"]) if not np.isnan(row["_jdriven"]) else 0.0
Jimp = float(row["_jimp"])
Jeq = float(row["_jeq"])

# --------------------- sección 2: inercia equivalente -----------------------
st.header("2) Inercia equivalente al eje del motor")

st.markdown(
    r"""
**Definición:**

\[
J_{\mathrm{eq}} \;=\; J_{m} \;+\; J_{\mathrm{driver}}
\;+\; \frac{J_{\mathrm{driven}} + J_{\mathrm{imp}}}{r^2}
\]
""",
)

st.caption(
    r"Las inercias del **lado bomba** giran a \(\omega_p = \omega_m / r\). "
    r"Igualando energías cinéticas a una \(\omega_m\) común se obtiene la división por \(r^2\) del término del lado bomba."
)

# Sustitución numérica renderizada
lhs = r"J_{\mathrm{eq}} \;=\; J_{m} + J_{\mathrm{driver}} + \dfrac{J_{\mathrm{driven}} + J_{\mathrm{imp}}}{r^2}"
nums = rf"= \; {Jm:.2f} \;+\; {Jdriver:.2f} \;+\; \dfrac{{{Jdriven:.2f} + {Jimp:.2f}}}{{({r:.2f})^2}}"
with st.container():
    c1, c2 = st.columns([3, 1.2])
    with c1:
        st.markdown("**Sustitución numérica:**")
        st.latex(lhs)
        st.latex(nums)
    with c2:
        st.metric(label="J_eq (kg·m²)", value=f"{Jeq:.2f}")

# --------------------- (opcional) gráfico de aporte de inercias --------------
if HAVE_PLOTLY:
    parts = ["J_m (motor)", "J_driver (polea+bushing motor)", "J_driven (polea+bushing bomba)", "J_imp (impulsor) reflejado"]
    values = [Jm, Jdriver, (Jdriven + Jimp) / (r**2), 0]  # ya incluimos J_imp reflejado dentro del mismo término
    # Mostrar J_imp reflejado aparte (para lectura):
    values[-1] = (Jimp) / (r**2)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=parts[:3], y=values[:3], name="Aporte directo"))
    fig.add_trace(go.Bar(x=[parts[3]], y=[values[3]], name="Solo J_imp reflejado"))
    fig.update_layout(barmode="group", title="Aporte de inercias en J_eq", yaxis_title="kg·m²")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Instala `plotly` para ver el gráfico de aportes de inercia.")


# --------------------- sección 3: comprobación rápida de n_torque -----------
st.header("3) Respuesta inercial (sin efectos hidráulicos)")

st.markdown(
    r"""
Definiciones:  
\(\dot n_{\mathrm{torque}}\) = tasa de aceleración por par \([\,\mathrm{rpm/s}\,]\),  
\(t\) = tiempo \([\,\mathrm{s}\,]\),  
\(\Delta n = n_f - n_i \ [\mathrm{rpm}]\),  
\(T_{\mathrm{disp}}\) = par disponible en el eje del motor \([\,\mathrm{Nm}\,]\).
"""
)

cA, cB, cC = st.columns(3)
with cA:
    n_i = st.number_input("Velocidad Motor inicial [rpm]", value=738.0, step=1.0, format="%.2f")
with cB:
    n_f = st.number_input("Velocidad Motor final [rpm]", value=1475.0, step=1.0, format="%.2f")
with cC:
    T_disp = st.number_input("Par disponible [Nm]", value=240.0, step=1.0, format="%.2f")

delta_n = n_f - n_i

# ramp VDF: fijo 300 rpm/s (puedes hacerlo configurable)
rampa_vdf = 300.0

# dot n torque
n_dot_torque = 60.0 * T_disp / (2 * np.pi * Jeq)  # rpm/s
t_par = delta_n / n_dot_torque
t_rampa = delta_n / rampa_vdf
t_final_sin = max(t_par, t_rampa)

st.latex(r"\dot n_{\mathrm{torque}} \;=\; \dfrac{60\,T_{\mathrm{disp}}}{2\pi\,J_{\mathrm{eq}}}")
st.latex(r"t_{\mathrm{par}} \;=\; \dfrac{\Delta n}{\dot n_{\mathrm{torque}}} \quad,\quad "
         r"t_{\mathrm{rampa}} \;=\; \dfrac{\Delta n}{\mathrm{rampa}_{\mathrm{VDF}}} \quad,\quad "
         r"t_{\mathrm{final,\,sin}} \;=\; \max\!\left(t_{\mathrm{par}},\,t_{\mathrm{rampa}}\right)")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Δn [rpm]", f"{delta_n:.2f}")
c2.metric("ṅ_torque [rpm/s]", f"{n_dot_torque:.2f}")
c3.metric("t_par [s]", f"{t_par:.2f}")
c4.metric("t_final (sin hidráulica) [s]", f"{t_final_sin:.2f}")

st.caption("Nota: en esta sección no se incluye aún el par hidráulico de la bomba ni la curva del sistema; sólo inercia mecánica y rampa del VDF.")


# -------------------- tabla resumen exportable por TAG -----------------------
st.header("Resumen por TAG (inercia)")

df_out = df[[col_tag, "_r", "_jm", "_jdriver", "_jdriven", "_jimp", "_jeq"]].copy()
df_out.columns = ["TAG", "r", "J_m_kgm2", "J_driver_kgm2", "J_driven_kgm2", "J_imp_kgm2", "J_eq_kgm2"]

st.dataframe(df_out.style.format({c: "{:.2f}" for c in df_out.columns if c != "TAG"}), use_container_width=True)

csv = df_out.to_csv(index=False).encode("utf-8")
st.download_button("Descargar tabla CSV", csv, file_name="resumen_inercias.csv", mime="text/csv")


# ---------------- depurado: muestra qué columnas se usaron -------------------
with st.expander("Columnas detectadas (debug)"):
    st.write({
        "TAG": col_tag,
        "relacion_transmision (r)": col_r,
        "J_m": col_jm,
        "J_driver_total": col_jdriver_total,
        "J_driver_pulley": col_jdriver_pul,
        "J_driver_bushing": col_jdriver_bush,
        "J_driven_total": col_jdriven_total,
        "J_driven_pulley": col_jdriven_pul,
        "J_driven_bushing": col_jdriven_bush,
        "J_imp": col_jimp,
        "Potencia (por si existe)": col_kw,
    })
