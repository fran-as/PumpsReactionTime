
# App Streamlit – Tiempo de reacción de bombas con VDF (MantoVerde)

## Cómo ejecutar
1) Instala dependencias (ideal en un entorno virtual):
   ```bash
   pip install streamlit pandas numpy matplotlib
   ```
2) Ejecuta la app:
   ```bash
   streamlit run streamlit_app_bombas_vdf.py
   ```

## ¿Qué hace?
- Carga una tabla de **entradas** (o usa un dataset por defecto).
- Permite editar **parámetros del VDF** (rampa), **sobrecarga de par (pu)** y **datos de motores** (T_nom, Jm).
- Calcula tiempos por **rampa** y por **par/inercia** (con inercia de impulsor como **disco**, **aro** o **rango**).
- Muestra **resultados** en tabla, permite **descargar CSV** y genera un **gráfico** de Δn vs. t por TAG.

## Columnas mínimas de entrada (CSV)
Application, TAG, P_inst [kW], P_eff [kW], Poles, n_motor_max [rpm], n_motor_min [rpm], r,
n_pump_max [rpm], n_pump_min [rpm], Ø_imp [mm], M_imp [kg]

## Notas
- El modelo actual es de **screening** (afinidad). Para precisión, integrar curvas bomba–sistema, límites de corriente del VDF y NPSH.
