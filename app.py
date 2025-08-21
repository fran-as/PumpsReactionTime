# app.py
# ──────────────────────────────────────────────────────────────────────────────
# Memoria de Cálculo – Tiempo de reacción de bombas (VDF)
# App “todo-en-uno” con:
#  1) Lectura del dataset `bombas_dataset_with_torque_params.xlsx`
#  2) Resumen de parámetros por TAG (motor, transmisión, bomba, sistema)
#  3) Respuesta inercial + VDF (fórmulas LaTeX y métricas clave)
#  4) Respuesta con hidráulica (modelo sencillo): integra J_eq·ω̇ = T_disp − T_pump/r
#  5) Gráfico interactivo (Plotly) con 3 ejes y descarga de reportes por bytes
# ──────────────────────────────────────────────────────────────────────────────
# =========================
# Loader de dataset fijo + mapeo a parámetros del modelo
# =========================
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

DATASET_PATH = "bombas_dataset_with_torque_params.xlsx"

# Conversión de inercia: 1 lb·ft² = 0.042140110093 kg·m²
LBFT2_TO_KGM2 = 0.042140110093

def get_num(x: object, default: float = 0.0) -> float:
    """Convierte valores con coma decimal, espacios, unidades sueltas, etc. a float."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return float(default)
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace(" ", "").replace("\u00a0", "")
    # coma decimal -> punto
    s = s.replace(",", ".")
    # quita cualquier cosa que no sea dígito, signo, punto o notación científica
    s = re.sub(r"[^0-9eE+\-\.]", "", s)
    try:
        return float(s) if s not in ("", ".", "-", "+") else float(default)
    except Exception:
        return float(default)

def as_int(x: object, default: int = 0) -> int:
    v = get_num(x, default=float(default))
    return int(round(v))

def j_lbft2_to_kgm2(v_lbft2: object) -> float:
    return get_num(v_lbft2) * LBFT2_TO_KGM2

def sanitize_eff(x: object) -> float:
    """
    Eficiencia en [0..1]. Acepta entrada 0.92, 92, '92%', '0,92', etc.
    - Si >1.1, se interpreta como % y se divide por 100.
    - Se recorta al rango [0, 1].
    """
    v = get_num(x)
    if v > 1.1:
        v = v / 100.0
    return float(min(max(v, 0.0), 1.0))

def calc_eta_total(row: pd.Series) -> float:
    # Usa 'eta_total' si existe y parece válida, si no, multiplica eta_a..eta_d
    if "eta_total" in row and not pd.isna(row["eta_total"]):
        v = sanitize_eff(row["eta_total"])
        if 0.01 <= v <= 1.0:
            return v
    # producto de eslabones (si alguna falta, get_num -> 0 y producto será 0; protegemos con default=1)
    parts = []
    for k in ("eta_a", "eta_b", "eta_c", "eta_d"):
        if k in row and not pd.isna(row[k]):
            parts.append(sanitize_eff(row[k]))
    return float(np.prod(parts)) if parts else 1.0

def calc_K_m_per_m3h2(row: pd.Series) -> float:
    """
    Devuelve K en unidades de m/(m^3/h)^2 para usar en: H(Q)=H0 + K * (Q/3600)^2
    Si hay 'K_m_per_m3h2', la usa; si no, convierte desde 'K_m_s2' multiplicando por 3600^2.
    """
    if "K_m_per_m3h2" in row and not pd.isna(row["K_m_per_m3h2"]):
        return get_num(row["K_m_per_m3h2"])
    # fallback desde K en m/(m^3/s)^2
    K_s2 = get_num(row.get("K_m_s2", 0.0))
    return K_s2 * (3600.0**2)

# -------------------------
# SCHEMA: Atributos del modelo ↔ columnas del dataset
# -------------------------
SCHEMA = {
    # Identificación / básicos
    "TAG":                 {"col": "TAG",              "dtype": "str",   "unit": "-",          "desc": "Identificador del equipo"},
    "brand":               {"col": "brand",            "dtype": "str",   "unit": "-",          "desc": "Marca (tren)"},
    "train":               {"col": "train",            "dtype": "str",   "unit": "-",          "desc": "Descripción del tren"},
    "motor_brand":         {"col": "motor_brand",      "dtype": "str",   "unit": "-",          "desc": "Marca motor"},
    "motor_model":         {"col": "motor_model",      "dtype": "str",   "unit": "-",          "desc": "Modelo motor"},
    "motorframe":          {"col": "motorframe",       "dtype": "str",   "unit": "-",          "desc": "Frame motor"},
    "pump_brand":          {"col": "pump_brand",       "dtype": "str",   "unit": "-",          "desc": "Marca bomba"},
    "pump_model":          {"col": "pump_model",       "dtype": "str",   "unit": "-",          "desc": "Modelo bomba"},

    # Transmisión / geometría (informativos y para chequeos)
    "r":                   {"col": "r_trans",          "dtype": "float", "unit": "-",          "desc": "Relación n_motor/n_bomba (transmisión)"},
    "series":              {"col": "series",           "dtype": "str",   "unit": "-",          "desc": "Perfil de correas (5V, etc.)"},
    "grooves":             {"col": "grooves",          "dtype": "int",   "unit": "-",          "desc": "N° de canales"},
    "driver_od_in":        {"col": "driver_od_in",     "dtype": "float", "unit": "in",         "desc": "Diámetro polea motriz [in]"},
    "driven_od_in":        {"col": "driven_od_in",     "dtype": "float", "unit": "in",         "desc": "Diámetro polea conducida [in]"},
    "centerdistance_mm":   {"col": "centerdistance_mm","dtype": "float", "unit": "mm",         "desc": "Distancia entre centros [mm]"},

    # Motor / eléctricos (informativo)
    "motor_kw":            {"col": "motor_kw",         "dtype": "float", "unit": "kW",         "desc": "Potencia nominal motor"},
    "motor_nom_rpm":       {"col": "motor_nom_rpm",    "dtype": "int",   "unit": "rpm",        "desc": "Velocidad nominal motor"},
    "motor_sf":            {"col": "motor_sf",         "dtype": "float", "unit": "-",          "desc": "Service Factor"},
    "motor_sf_torque":     {"col": "motor_sf_torque",  "dtype": "float", "unit": "Nm",         "desc": "Torque con SF"},
    "motor_rated_torque":  {"col": "motor_rated_torque","dtype":"float", "unit": "Nm",         "desc": "Torque nominal"},
    "motor_rated_current": {"col": "motor_rated_current","dtype":"float","unit": "A",          "desc": "Corriente nominal"},
    "motor_rating":        {"col": "motor_rating",     "dtype": "str",   "unit": "-",          "desc": "Grado de protección / clase"},

    # Límites de operación (se usan en el modelo)
    "n_motor_min":         {"col": "motor_n_min_rpm",  "dtype": "int",   "unit": "rpm",        "desc": "n motor mín"},
    "n_motor_max":         {"col": "motor_n_max_rpm",  "dtype": "int",   "unit": "rpm",        "desc": "n motor máx"},
    "n_pump_min":          {"col": "pump_n_min_rpm",   "dtype": "int",   "unit": "rpm",        "desc": "n bomba mín"},
    "n_pump_max":          {"col": "pump_n_max_rpm",   "dtype": "int",   "unit": "rpm",        "desc": "n bomba máx"},

    # Hidráulica (SE USAN en la curva H(Q)=H0 + K*(Q/3600)^2 )
    "H0_m":                {"col": "H0_m",             "dtype": "float", "unit": "m",          "desc": "Altura a caudal cero (shutoff)"},
    # 'K' la derivamos con calc_K_m_per_m3h2; lo dejamos en el diccionario final
    "R2_H":                {"col": "R2_H",             "dtype": "float", "unit": "-",          "desc": "R² ajuste H(Q)"},
    "eta":                 {"col": "eta_total",        "dtype": "float", "unit": "-",          "desc": "Eficiencia total (si falta, eta_a*eta_b*eta_c*eta_d)"},

    # Inercias (SE USAN en el modelo)
    "J_m":                 {"col": "motor_j_kgm2",         "dtype": "float", "unit": "kg·m²", "desc": "Inercia motor"},
    "J_driver_pulley":     {"col": "driverpulley_j_kgm2",  "dtype": "float", "unit": "kg·m²", "desc": "Inercia polea motriz"},
    "J_driver_bushing":    {"col": "driverbushing_j_kgm2", "dtype": "float", "unit": "kg·m²", "desc": "Inercia buje motriz"},
    "J_driven_pulley_lbft2":{"col":"drivenpulley_j_lbs_ft2","dtype":"float","unit":"lb·ft²", "desc": "Inercia polea conducida (imperial)"},
    "J_driven_bushing_lbft2":{"col":"drivenbushing_j_lbs_ft2","dtype":"float","unit":"lb·ft²","desc":"Inercia buje conducido (imperial)"},
    "J_imp":               {"col": "impeller_j_kgm2",      "dtype": "float", "unit": "kg·m²", "desc": "Inercia rodete"},
    # Extras informativos
    "impeller_mass_kg":    {"col": "impeller_mass_kg",     "dtype": "float", "unit": "kg",    "desc": "Masa del rodete"},
}

def normalize_scalar(colname: str, raw: object, dtype: str):
    """Normaliza un valor de acuerdo con el dtype declarado en SCHEMA."""
    if dtype == "str":
        return "" if (raw is None or (isinstance(raw, float) and np.isnan(raw))) else str(raw).strip()
    if dtype == "int":
        return as_int(raw)
    if dtype == "float":
        return get_num(raw)
    return raw

def build_row_params(row: pd.Series) -> dict:
    """
    Devuelve los parámetros normalizados que usa el modelo para un TAG.
    Incluye: r, H0_m, K, eta, inercia desagregada y compuesta, límites de rpm y meta-info útil.
    """
    out = {}

    # 1) Campos directos del SCHEMA
    for key, meta in SCHEMA.items():
        col = meta["col"]
        dtype = meta["dtype"]
        out[key] = normalize_scalar(col, row.get(col, None), dtype)

    # 2) Derivados / limpiezas especiales

    # Relación r (usa r_trans)
    out["r"] = get_num(row.get("r_trans", out.get("r", 0.0)))

    # Curva H(Q): H0 directo + K en m/(m^3/h)^2
    out["K"] = calc_K_m_per_m3h2(row)

    # Eficiencia total
    out["eta"] = calc_eta_total(row)

    # Inercias compuestas
    #  - Lado motor (driver), ya viene en kg·m²
    J_driver = get_num(row.get("driverpulley_j_kgm2", 0.0)) + get_num(row.get("driverbushing_j_kgm2", 0.0))
    #  - Lado bomba (driven), viene en lb·ft² -> convertir
    J_driven = j_lbft2_to_kgm2(row.get("drivenpulley_j_lbs_ft2", 0.0)) + j_lbft2_to_kgm2(row.get("drivenbushing_j_lbs_ft2", 0.0))

    out["J_driver"] = float(J_driver)
    out["J_driven"] = float(J_driven)

    # 3) Paquete final que consume el modelo (nombres "canónicos" usados en la app)
    params_modelo = {
        "TAG": out["TAG"],
        # Geometría / transmisión
        "r": out["r"],

        # Curva H(Q) con Q en m^3/h
        "H0_m": out["H0_m"],
        "K": out["K"],            # m/(m^3/h)^2
        "R2_H": out["R2_H"],

        # Eficiencia global
        "eta": out["eta"],        # [0..1]

        # Inercias (kg·m²)
        "J_m": out["J_m"],
        "J_driver": out["J_driver"],
        "J_driven": out["J_driven"],
        "J_imp": out["J_imp"],

        # Límites de operación
        "n_motor_min": out["n_motor_min"],
        "n_motor_max": out["n_motor_max"],
        "n_pump_min":  out["n_pump_min"],
        "n_pump_max":  out["n_pump_max"],

        # Meta-info útil para UI / reporte
        "_meta": {
            "brand": out.get("brand", ""),
            "train": out.get("train", ""),
            "motor": {
                "brand": out.get("motor_brand", ""),
                "model": out.get("motor_model", ""),
                "frame": out.get("motorframe", ""),
                "kw": out.get("motor_kw", 0.0),
                "nom_rpm": out.get("motor_nom_rpm", 0),
                "sf": out.get("motor_sf", 0.0),
                "sf_torque": out.get("motor_sf_torque", 0.0),
                "rated_torque": out.get("motor_rated_torque", 0.0),
                "rated_current": out.get("motor_rated_current", 0.0),
                "rating": out.get("motor_rating", ""),
            },
            "pump": {
                "brand": out.get("pump_brand", ""),
                "model": out.get("pump_model", ""),
                "impeller_mass_kg": out.get("impeller_mass_kg", 0.0),
            },
            "belt_drive": {
                "series": out.get("series", ""),
                "grooves": out.get("grooves", 0),
                "driver_od_in": out.get("driver_od_in", 0.0),
                "driven_od_in": out.get("driven_od_in", 0.0),
                "center_mm": out.get("centerdistance_mm", 0.0),
            },
        },
    }

    # Validaciones suaves (evitan NaNs en el modelo)
    for k in ("H0_m", "K", "eta", "J_m", "J_driver", "J_driven", "J_imp", "r"):
        if not np.isfinite(params_modelo[k]):
            params_modelo[k] = 0.0
    for k in ("n_motor_min", "n_motor_max", "n_pump_min", "n_pump_max"):
        if not isinstance(params_modelo[k], (int, np.integer)):
            params_modelo[k] = as_int(params_modelo[k])

    return params_modelo

def load_pumps_db(xlsx_path: str = DATASET_PATH) -> dict[str, dict]:
    """Lee el Excel fijo y devuelve {TAG: params_modelo_normalizados}."""
    # dtype=object para no perder cadenas con coma decimal
    df = pd.read_excel(xlsx_path, sheet_name="dataset", dtype=object)

    # Normaliza nombres de columnas esperadas (por si vienen con espacios raros)
    df.columns = [str(c).strip() for c in df.columns]

    pumps = {}
    for _, row in df.iterrows():
        tag = str(row.get("TAG", "")).strip()
        if not tag:
            continue
        pumps[tag] = build_row_params(row)

    return pumps

# Construimos el diccionario AL ARRANCAR
PUMPS_DB = load_pumps_db()

# =========================
# Ejemplo de uso en la app:
#   tags = sorted(PUMPS_DB.keys())
#   tag = st.selectbox("TAG", tags)
#   params = PUMPS_DB[tag]
#   # Luego usar params["H0_m"], params["K"], params["r"], params["eta"], etc.
# =========================



# Plotly (si fallara la import, mostramos un mensaje y seguimos con cálculo)
try:
    import plotly.graph_objects as go
except Exception as e:
    go = None

# ──────────────────────────────────────────────────────────────────────────────
# Configuración de página
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Memoria de Cálculo – Tiempo de reacción (VDF)",
    page_icon="⏱️",
    layout="wide",
)

st.title("⏱️ Tiempo de reacción de bombas con VDF")
st.write(
    "Herramienta para estimar tiempos de reacción considerando inercia, rampa VDF "
    "y carga hidráulica simplificada."
)

# ──────────────────────────────────────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────────────────────────────────────

def get_num(x: Any) -> float:
    """Convierte texto/num a float de forma robusta (admite coma decimal, espacios, etc.)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0.0
    if isinstance(x, (int, float, np.floating)):
        return float(x)
    s = str(x).strip().replace(" ", "").replace("\u00a0", "")
    s = s.replace(",", ".")
    s = re.sub(r"[^\d\.\-eE+]", "", s)
    try:
        return float(s)
    except Exception:
        return 0.0


def csv_bytes_from_df(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def show_badge(value_text: str, unit: str, label: str) -> None:
    """Badge básico en HTML (inline)."""
    st.markdown(
        f"""
        <div style="
            display:inline-block;
            padding:6px 10px;
            margin:4px 6px 4px 0;
            border-radius:12px;
            background:#f0f2f6;
            border:1px solid #e0e3e8;
            font-size:0.9rem;">
            <strong>{label}:</strong> {value_text} <span style="opacity:.7">{unit}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


@dataclass
class TagParams:
    TAG: str
    # Motor / transmisión / bomba
    J_m: float           # [kg·m²] Inercia motor
    J_driver: float      # [kg·m²] Inercia transmisión (acopl./reductor, lado motor)
    J_driven: float      # [kg·m²] Inercia eje bomba (lado bomba)
    J_imp: float         # [kg·m²] Inercia impulsor adicional (lado bomba)
    n_motor_nom: float   # [rpm] nominal motor
    n_bomba_nom: float   # [rpm] nominal bomba
    P_motor_kw: float    # [kW] potencia nominal motor (para estimar T_disp si no hay)
    T_disp: float        # [Nm] par disponible (constante simplificado)
    # Curva bomba y sistema
    Q_nom_m3h: float     # [m³/h] caudal nominal
    H0_m: float          # [m] intersección con eje H
    K: float             # [-] coef. cuadrático (si Q en m³/h, ver fórmula)
    eta_h: float         # [-] eficiencia hidráulica global
    rho_kgm3: float      # [kg/m³] densidad
    Q_min_m3h: float     # [m³/h] clamp inferior
    Q_max_m3h: float     # [m³/h] clamp superior

    @property
    def r(self) -> float:
        """Relación r = n_motor / n_bomba."""
        n_b = self.n_bomba_nom if self.n_bomba_nom > 0 else 1.0
        return max(self.n_motor_nom, 1e-6) / n_b

    @property
    def J_eq(self) -> float:
        """Inercia equivalente referida al eje del motor."""
        return self.J_m + self.J_driver + (self.J_driven + self.J_imp) / max(self.r**2, 1e-9)


# ──────────────────────────────────────────────────────────────────────────────
# Carga de datos
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_default_df() -> pd.DataFrame:
    # DF de ejemplo si el archivo no está; columnas esperadas.
    data = [
        {
            "TAG": "PU-101",
            "J_m": 0.45,
            "J_driver": 0.05,
            "J_driven": 0.30,
            "J_imp": 0.15,
            "n_motor_nom": 1500.0,
            "n_bomba_nom": 1000.0,
            "P_motor_kw": 90.0,
            "T_disp": 0.0,  # 0 => estimar desde P y n
            "Q_nom_m3h": 250.0,
            "H0_m": 30.0,
            "K": 0.002,     # si Q está en m³/h se usa (Q/3600)^2
            "eta_h": 0.72,
            "rho_kgm3": 1000.0,
            "Q_min_m3h": 20.0,
            "Q_max_m3h": 450.0,
        },
        {
            "TAG": "PU-102",
            "J_m": 0.60,
            "J_driver": 0.06,
            "J_driven": 0.28,
            "J_imp": 0.12,
            "n_motor_nom": 1800.0,
            "n_bomba_nom": 1200.0,
            "P_motor_kw": 110.0,
            "T_disp": 0.0,
            "Q_nom_m3h": 300.0,
            "H0_m": 28.0,
            "K": 0.0025,
            "eta_h": 0.70,
            "rho_kgm3": 1000.0,
            "Q_min_m3h": 30.0,
            "Q_max_m3h": 500.0,
        },
    ]
    return pd.DataFrame(data)


@st.cache_data
def load_dataset(upload: io.BytesIO | None) -> pd.DataFrame:
    # Si subieron archivo, lo usamos; si no, intentamos ruta por defecto; si no, DF ejemplo.
    if upload is not None:
        try:
            return pd.read_excel(upload)
        except Exception:
            upload.seek(0)
            try:
                return pd.read_csv(upload)
            except Exception:
                pass

    # Intento de ruta por defecto del repo
    try:
        return pd.read_excel("bombas_dataset_with_torque_params.xlsx")
    except Exception:
        return load_default_df()


def row_to_params(row: pd.Series) -> TagParams:
    # Obtención robusta de columnas con defaults si no existen.
    def gv(col: str, default: float = 0.0) -> float:
        return get_num(row[col]) if col in row else default

    return TagParams(
        TAG=str(row.get("TAG", "SIN_TAG")),
        J_m=gv("J_m"),
        J_driver=gv("J_driver"),
        J_driven=gv("J_driven"),
        J_imp=gv("J_imp"),
        n_motor_nom=max(gv("n_motor_nom", 1500.0), 1e-6),
        n_bomba_nom=max(gv("n_bomba_nom", 1000.0), 1e-6),
        P_motor_kw=max(gv("P_motor_kw", 75.0), 0.0),
        T_disp=gv("T_disp"),  # si 0 -> estimar
        Q_nom_m3h=max(gv("Q_nom_m3h", 200.0), 1e-9),
        H0_m=gv("H0_m", 25.0),
        K=gv("K", 0.002),
        eta_h=min(max(gv("eta_h", 0.70), 0.05), 0.95),
        rho_kgm3=max(gv("rho_kgm3", 1000.0), 1.0),
        Q_min_m3h=max(gv("Q_min_m3h", 10.0), 0.0),
        Q_max_m3h=max(gv("Q_max_m3h", 600.0), 0.0),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Entradas")
    up = st.file_uploader("Subir dataset (.xlsx o .csv)", type=["xlsx", "csv"])
    df = load_dataset(up)

    if "TAG" not in df.columns:
        st.error("El dataset debe tener una columna `TAG`.")
        st.stop()

    tag_list = sorted(df["TAG"].astype(str).unique().tolist())
    sel_tag = st.selectbox("TAG", options=tag_list, index=0)

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        n_ini = st.number_input("n inicial bomba [rpm]", value=0.0, min_value=0.0, step=10.0)
    with col_r2:
        n_fin = st.number_input("n objetivo bomba [rpm]", value=1000.0, min_value=1.0, step=10.0)

    rampa_vdf = st.slider("Rampa VDF [rpm/s] (motor)", min_value=1, max_value=500, value=200, step=5)

    t_max = st.slider("t_max integración [s]", min_value=1, max_value=120, value=30, step=1)
    dt = st.select_slider("dt [s]", options=[0.001, 0.002, 0.005, 0.01, 0.02], value=0.01)


# ──────────────────────────────────────────────────────────────────────────────
# Parámetros del TAG seleccionado
# ──────────────────────────────────────────────────────────────────────────────
row = df[df["TAG"].astype(str) == str(sel_tag)].iloc[0]
params = row_to_params(row)

st.subheader(f"📌 Parámetros – **{params.TAG}**")

# Ecuación J_eq
st.latex(r"J_{\mathrm{eq}} = J_m + J_{\mathrm{driver}} + \dfrac{J_{\mathrm{driven}}+J_{\mathrm{imp}}}{r^2}")
st.caption(r"$r = n_{\mathrm{motor}}/n_{\mathrm{bomba}}$; las inercias del lado bomba giran a $\omega_p=\omega_m/r$.")

# Badges (formateo correcto del H0_m como string)
c1, c2, c3, c4 = st.columns(4)
with c1:
    show_badge(f"{params.J_eq:.3f}", "kg·m²", "J_eq")
    show_badge(f"{params.r:.2f}", "", "Relación r")
with c2:
    show_badge(f"{params.n_motor_nom:.0f}", "rpm", "n_motor_nom")
    show_badge(f"{params.n_bomba_nom:.0f}", "rpm", "n_bomba_nom")
with c3:
    show_badge(f"{params.Q_nom_m3h:.1f}", "m³/h", "Q_nom")
    show_badge(f"{params.H0_m:.2f}", "m", "H0")
with c4:
    show_badge(f"{params.K:.4f}", "–", "K")
    show_badge(f"{params.eta_h:.2f}", "–", "η")

# ──────────────────────────────────────────────────────────────────────────────
# Métricas inerciales (sin hidráulica): t_par y t_rampa
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("### ⚡ Estimación inercial vs rampa VDF")

# Estimar T_disp si es 0: T = 9550 * P[kW] / n[rpm] a nominal motor
T_disp = params.T_disp
if T_disp <= 0.0:
    T_disp = 9550.0 * max(params.P_motor_kw, 1e-9) / max(params.n_motor_nom, 1e-6)

omega_dot_torque = (T_disp / max(params.J_eq, 1e-9))  # [rad/s²] sobre el eje motor
# Pasar a \dot n motor [rpm/s]: \dot n = (60/2π)·\dot ω
n_dot_torque_motor = (60.0 / (2.0 * np.pi)) * omega_dot_torque

# Δn motor equivalente del salto en n bomba: Δn_motor = r * Δn_bomba
delta_n_bomba = max(n_fin - n_ini, 0.0)
delta_n_motor = params.r * delta_n_bomba

t_par = delta_n_motor / max(n_dot_torque_motor, 1e-9)         # limitado por par
t_rampa = delta_n_motor / max(float(rampa_vdf), 1e-9)          # limitado por rampa VDF
t_inercial_aprox = max(t_par, t_rampa)

mcol1, mcol2, mcol3 = st.columns(3)
with mcol1:
    st.metric("t_par (solo par)", f"{t_par:.2f} s")
with mcol2:
    st.metric("t_rampa (VDF)", f"{t_rampa:.2f} s")
with mcol3:
    st.metric("t_reacción aprox", f"{t_inercial_aprox:.2f} s")

st.caption(
    r"Ecuaciones: $\dot n_{\mathrm{torque}}=\frac{60}{2\pi}\frac{T_{\mathrm{disp}}}{J_{\mathrm{eq}}}$, "
    r"$t_{\mathrm{par}}=\frac{\Delta n_{\mathrm{motor}}}{\dot n_{\mathrm{torque}}}$, "
    r"$t_{\mathrm{rampa}}=\frac{\Delta n_{\mathrm{motor}}}{\mathrm{rampa}_{\mathrm{VDF}}}$."
)

# ──────────────────────────────────────────────────────────────────────────────
# Dinámica con hidráulica (modelo simple)
# J_eq·ω̇_m = T_disp − T_pump/r
# con T_pump = ρgQH(Q)/(η ω_p), ω_p=ω_m/r, Q ∝ n_p (afinidad), clamp [Qmin, Qmax]
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("🌊 Integración con hidráulica (modelo simple)")

st.latex(
    r"J_{eq}\,\dot\omega_m = T_{disp} - \dfrac{T_{pump}}{r}, \qquad "
    r"T_{pump}=\dfrac{\rho\,g\,Q\,H(Q)}{\eta\,\omega_p}, \; \omega_p=\omega_m/r, \; Q\propto n_p."
)
st.caption(
    r"Se impone además un límite de rampa VDF en el motor: $n_m(t)$ no puede crecer "
    r"más rápido que $\mathrm{rampa}_{\mathrm{VDF}}$."
)

g = 9.81

def H_of_Q(Q_m3h: float, H0: float, K: float) -> float:
    """Curva H(Q) = H0 + K*(Q/3600)^2 con Q en m³/h, devuelve H en m."""
    return H0 + K * (Q_m3h / 3600.0) ** 2

def simulate(params: TagParams,
             n_ini_bomba: float,
             n_fin_bomba: float,
             rampa_vdf_motor: float,
             t_max: float,
             dt: float,
             T_disp_const: float) -> Dict[str, np.ndarray]:
    steps = int(np.ceil(t_max / dt))
    t = np.zeros(steps + 1)
    n_b = np.zeros(steps + 1)      # rpm bomba actual
    n_m = np.zeros(steps + 1)      # rpm motor actual
    n_cmd_m = np.zeros(steps + 1)  # rpm motor comandada por VDF
    Q = np.zeros(steps + 1)        # m³/h
    PkW = np.zeros(steps + 1)      # kW hidráulica

    n_b[0] = max(n_ini_bomba, 0.0)
    n_m[0] = params.r * n_b[0]
    n_cmd_m[0] = n_m[0]

    n_fin_m = params.r * max(n_fin_bomba, 1.0)

    for k in range(steps):
        t[k + 1] = t[k] + dt

        # VDF: referencia de velocidad motor por rampa
        inc = rampa_vdf_motor * dt
        n_cmd_m[k + 1] = min(n_cmd_m[k] + inc, n_fin_m)

        # Estado actual
        omega_m = 2.0 * np.pi * n_m[k] / 60.0
        omega_p = max(omega_m / max(params.r, 1e-9), 1e-6)

        # Afinidad Q ~ n_bomba
        n_b[k] = max(n_b[k], 0.0)
        Q_k = params.Q_nom_m3h * (n_b[k] / max(params.n_bomba_nom, 1e-6))
        # Clamp hidráulico
        Q_k = min(max(Q_k, params.Q_min_m3h), params.Q_max_m3h)

        # Curva de bomba
        H = H_of_Q(Q_k, params.H0_m, params.K)

        # Par de bomba (lado bomba): T_pump = ρ g Q H / (η ω_p)
        Q_m3s = Q_k / 3600.0
        T_pump = (params.rho_kgm3 * g * Q_m3s * H) / max(params.eta_h * omega_p, 1e-6)

        # Dinámica en el eje motor
        domega_m = (T_disp_const - (T_pump / max(params.r, 1e-9))) / max(params.J_eq, 1e-9)
        omega_m_new = max(omega_m + domega_m * dt, 0.0)
        n_m_candidate = 60.0 * omega_m_new / (2.0 * np.pi)

        # Limitación por rampa (nunca superar n_cmd_m)
        n_m[k + 1] = min(n_m_candidate, n_cmd_m[k + 1])
        n_b[k + 1] = n_m[k + 1] / max(params.r, 1e-9)

        # Potencia hidráulica
        Q_k_next = params.Q_nom_m3h * (n_b[k + 1] / max(params.n_bomba_nom, 1e-6))
        Q_k_next = min(max(Q_k_next, params.Q_min_m3h), params.Q_max_m3h)
        H_next = H_of_Q(Q_k_next, params.H0_m, params.K)
        PkW[k + 1] = (params.rho_kgm3 * g * (Q_k_next / 3600.0) * H_next) / 1000.0

        Q[k + 1] = Q_k_next

        # Si ya alcanzamos n_fin_m (tolerancia), podemos cortar
        if n_m[k + 1] >= n_fin_m - 1e-6 and n_cmd_m[k + 1] >= n_fin_m - 1e-6:
            # recortar arrays
            last = k + 1
            t = t[: last + 1]
            n_b = n_b[: last + 1]
            n_m = n_m[: last + 1]
            n_cmd_m = n_cmd_m[: last + 1]
            Q = Q[: last + 1]
            PkW = PkW[: last + 1]
            break

    return {"t": t, "n_b": n_b, "n_m": n_m, "n_cmd_m": n_cmd_m, "Q": Q, "PkW": PkW}


sim = simulate(
    params=params,
    n_ini_bomba=n_ini,
    n_fin_bomba=n_fin,
    rampa_vdf_motor=float(rampa_vdf),
    t_max=float(t_max),
    dt=float(dt),
    T_disp_const=float(T_disp),
)

# ──────────────────────────────────────────────────────────────────────────────
# Gráfico Plotly con 3 ejes corregidos (sin error de position fuera de [0,1])
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("#### 📈 Curvas de arranque")

if go is None:
    st.warning("Plotly no está disponible. Se omitirá el gráfico interactivo.")
else:
    fig = go.Figure()
    t = sim["t"]
    Q = sim["Q"]
    n_p = sim["n_b"]
    P = sim["PkW"]

    fig.add_trace(go.Scatter(x=t, y=Q, name="Q [m³/h]", yaxis="y"))
    fig.add_trace(go.Scatter(x=t, y=n_p, name="n bomba [rpm]", yaxis="y2"))
    fig.add_trace(go.Scatter(x=t, y=P, name="P hidráulica [kW]", yaxis="y3"))
    fig.add_trace(go.Scatter(x=t, y=sim["n_cmd_m"] / max(params.r, 1e-9),
                             name="n bomba cmd [rpm]", yaxis="y2",
                             line=dict(dash="dash")))

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        xaxis=dict(title="Tiempo [s]"),

        # Eje izquierdo (Q)
        yaxis=dict(title="Q [m³/h]", anchor="x"),

        # Eje derecho (n)
        yaxis2=dict(title="n bomba [rpm]", overlaying="y", side="right", anchor="x"),

        # Tercer eje a la derecha con anchor free y position < 1 (FIX)
        yaxis3=dict(title="P [kW]", overlaying="y", side="right", anchor="free", position=0.98),
    )
    st.plotly_chart(fig, use_container_width=True)

st.caption(
    r"Ecuaciones: $J_{eq}=J_m+J_{driver}+(J_{driven}+J_{imp})/r^2$, "
    r"$T_{pump}=\frac{\rho g Q H(Q)}{\eta\,\omega_p}$, con $Q\propto n_p$ y "
    r"$H(Q)=H_0+K\left(\frac{Q}{3600}\right)^2$."
)

# ──────────────────────────────────────────────────────────────────────────────
# Reportes y descargas (por BYTES, sin rutas -> evita MediaFileHandler)
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("📥 Reportes")

# Reporte del TAG actual (resumen + última simulación)
rep_tag = pd.DataFrame(
    {
        "TAG": [params.TAG],
        "J_eq_kgm2": [params.J_eq],
        "r": [params.r],
        "T_disp_Nm": [T_disp],
        "n_ini_bomba_rpm": [n_ini],
        "n_fin_bomba_rpm": [n_fin],
        "rampa_vdf_motor_rpmps": [rampa_vdf],
        "t_par_s": [t_par],
        "t_rampa_s": [t_rampa],
        "t_reaccion_aprox_s": [max(t_par, t_rampa)],
    }
)

csv_one = csv_bytes_from_df(rep_tag)

st.download_button(
    "⬇️ Descargar reporte del TAG seleccionado",
    data=csv_one,
    file_name=f"reporte_{params.TAG}_rampa_{int(rampa_vdf)}rpmps.csv",
    mime="text/csv",
)

# Reporte para TODOS los TAG con la rampa actual
def build_all_tags_report(df: pd.DataFrame, rampa: float) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        p = row_to_params(r)
        # Estimar T_disp si no hay
        T = p.T_disp if p.T_disp > 0 else 9550.0 * max(p.P_motor_kw, 1e-9) / max(p.n_motor_nom, 1e-6)
        omega_dot = T / max(p.J_eq, 1e-9)
        n_dot = (60.0 / (2.0 * np.pi)) * omega_dot
        # Δn_motor equivalente para 0 -> n_bomba_nom (referencia común)
        dnm = p.r * max(p.n_bomba_nom, 1.0)
        t_par_i = dnm / max(n_dot, 1e-9)
        t_rampa_i = dnm / max(rampa, 1e-9)
        rows.append(
            {
                "TAG": p.TAG,
                "J_eq_kgm2": p.J_eq,
                "r": p.r,
                "T_disp_Nm": T,
                "n_obj_bomba_rpm": p.n_bomba_nom,
                "rampa_vdf_motor_rpmps": rampa,
                "t_par_s": t_par_i,
                "t_rampa_s": t_rampa_i,
                "t_reaccion_aprox_s": max(t_par_i, t_rampa_i),
                "H0_m": p.H0_m,
                "K": p.K,
                "Q_nom_m3h": p.Q_nom_m3h,
            }
        )
    return pd.DataFrame(rows)

rep_all = build_all_tags_report(df, float(rampa_vdf))
csv_all = csv_bytes_from_df(rep_all)

st.download_button(
    "⬇️ Descargar reporte (todos los TAG, con rampa seleccionada)",
    data=csv_all,
    file_name=f"reporte_todos_los_TAG_rampa_{int(rampa_vdf)}rpmps.csv",
    mime="text/csv",
)

st.markdown("---")
st.markdown("#### 📝 Notas")
st.markdown(
    r"""
- El modelo hidráulico es intencionalmente simple y usa $Q\propto n_p$ y $H(Q)=H_0+K\left(\frac{Q}{3600}\right)^2$.
- Si la lectura del dataset trae celdas con texto (comas, unidades), se parsean con una rutina robusta.
- Los botones de descarga generan **bytes en memoria** (no rutas), evitando errores `MediaFileHandler`.
- El gráfico Plotly usa un **tercer eje** en `position=0.98` (dentro de [0,1]) para evitar el error de `layout.yaxis.position`.
"""
)
