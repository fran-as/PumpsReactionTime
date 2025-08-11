
# App Streamlit – Tiempo de reacción de bombas con VDF (MantoVerde)

## 🚀 Objetivo
Modelar el **tiempo de reacción** (aceleración/deceleración) de bombas centrífugas con VDF ante cambios de consigna, combinando:
- **Rampa del VDF** (límite de velocidad de cambio en el motor) y  
- **Dinámica por par e inercia** (torque disponible vs. torque de carga y momento de inercia equivalente).

La app permite **editar parámetros**, **cargar tus datos**, **ver resultados** y **descargar CSV**.

---

## 📦 Requisitos
```bash
python 3.9+
pip install streamlit pandas numpy matplotlib
```

---

## ▶️ Cómo ejecutar
1) Descarga los archivos:
   - `streamlit_app_bombas_vdf.py`

2) Ejecuta:
```bash
streamlit run streamlit_app_bombas_vdf.py
```

---

## 🧭 Uso de la app
1. **Datos de entrada**
   - Sube un **CSV** con las columnas mínimas (ver abajo) o usa el dataset por defecto.
   - Puedes **editar** la tabla en pantalla.
   - Botón para **descargar** las entradas en CSV.

2. **Parámetros (sidebar)**
   - **Rampa VDF (rpm/s motor)**: por defecto 300.
   - **Sobrecarga de par (pu)**: por defecto 1.0 (puedes probar 1.2–1.5).
   - **Modelo de inercia del impulsor**: *disc* (disco), *ring* (aro), o *range* (ambos).
   - **Datos de motor por TAG** (`T_nom`, `Jm`, `n_nom`) editables.

3. **Resultados**
   - Muestra **tiempos por rampa** y **por par/inercia** (disc/ring), y el **tiempo final** (el que manda).
   - Puedes **descargar** los resultados en CSV.
   - Gráfico Δn vs t por rampa para un TAG seleccionado.

---

## 🧠 Modelo (enfoque)
1. **Afinidad (screening)**  
   \\(Q \\propto n\\), \\(H \\propto n^2\\), \\(P \\propto n^3\\) ⇒ \\(T_{load} \\propto n^2\\).

2. **Inercia equivalente al eje del motor**  
   \\(J_{eq} = J_m + r^2\\,J_p\\), con \\(r = n_{motor}/n_{bomba}\\).  
   \\(J_p\\) se acota entre:
   - **Disco sólido**: \\(J = \\tfrac{1}{2} m R^2\\)  
   - **Aro delgado**: \\(J = m R^2\\)

3. **Rampa del VDF**  
   Aceleración en bomba = \\(\\text{rampa}_{motor}/r\\).  
   \\(t_{\\text{rampa}} = \\Delta n_{bomba} / (\\text{rampa}_{motor}/r)\\).

4. **Limitación por par/inercia**  
   \\(a_{par} = \\dfrac{T_{avail} - T_{load}}{J_{eq}}\\) (en rad/s², convertido a rpm/s).  
   Se integra numéricamente de \\(n_{motor,min}\\) a \\(n_{motor,max}\\) y se toma el **mínimo** entre \\(a_{par}\\) y la **rampa del VDF** en cada paso.

5. **Tiempo final por TAG**  
   \\(t_{\\text{final}} = \\max(t_{\\text{rampa}}, t_{\\text{par/inercia}})\\).

---

## 📌 Supuestos
- Zona de **par constante** hasta velocidad nominal (sin debilitamiento de campo).
- **Sobrecarga de par** configurable; el límite real puede estar dado por el VDF/corriente.
- No se consideran (por ahora) inercias de **acoples/poleas** ni **curva del sistema**; la carga se aproxima por afinidad.
- En **deceleración** podría requerirse **freno dinámico** para cumplir tiempos sin sobrevoltaje del bus DC.

---

## 📥 Columnas mínimas del CSV de entrada
```
Application
TAG
P_inst [kW]
P_eff [kW]
Poles
n_motor_max [rpm]
n_motor_min [rpm]
r
n_pump_max [rpm]
n_pump_min [rpm]
Ø_imp [mm]
M_imp [kg]
```

> **Datos de motor por TAG** (en el sidebar):  
> `T_nom` (Nm), `Jm` (kg·m²) y opcionalmente `n_nom` (rpm).

---

## 📤 Salidas
- **Tabla de resultados**:  
  - `Pump accel VFD-only [rpm/s]`  
  - `t_ramp_only [s]`  
  - `t_disc [s]`, `t_ring [s]`  
  - `t_final_min [s]`, `t_final_max [s]` (modo *range*) o `t_final [s]` (modo *disc/ring*)
- **Descarga CSV** de resultados y entradas.
- **Gráfico** Δn vs t por rampa para un TAG.

---

## ✅ Conclusiones preliminares
- El **mínimo teórico** está acotado por la **rampa del VDF** (importan \\(\\Delta n\\) y \\(r\\)).
- Si el **par disponible** es bajo y/o \\(J_{eq}\\) alto, el tiempo **real** lo domina el **par/inercia**.
- Recomendado definir **rampas diferenciadas** (subida/bajada) y validar **corriente** y **NPSH** para puntos acelerados.

---

## 🔧 Próximas mejoras (info solicitada)
- **Curva bomba + curva del sistema** por TAG para estimar \\(T_{load}(n)\\) real (TDH, SG, FF).
- **Límites de corriente y sobrecarga temporal** del VDF (por TAG).
- **Inercia de transmisión** (poleas/acoples) y **inercia real del impulsor** del fabricante.
- **Volumen/setpoints** de cada **cajón** para traducir \\(\\Delta Q\\) a **respuesta de nivel** y sintonía del control.

---

## 📁 Archivos
- App Streamlit: `streamlit_app_bombas_vdf.py`
