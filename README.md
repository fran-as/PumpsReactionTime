# App Streamlit – Tiempo de reacción de bombas con VDF (MantoVerde)

Incluye:
- **Fórmulas en LaTeX** (legibles).
- Cálculo de **r** desde **diámetros de poleas** (con ajuste por slip).
- **Modelo de curva de sistema** `H = H0 + K·Q²` → `T_load(n)` físico.
- Tabla editable, descarga de entradas y resultados, y gráfico Δn–t por TAG.

## Requisitos
```bash
python 3.9+
pip install -r requirements.txt
```

## Ejecutar
```bash
streamlit run streamlit_app_bombas_vdf_v2.py
```

## CSV de entrada mínimo
Application,TAG,P_inst [kW],P_eff [kW],Poles,n_motor_max [rpm],n_motor_min [rpm],r,n_pump_max [rpm],n_pump_min [rpm],Ø_imp [mm],M_imp [kg]

## Curvas de sistema
En el **expander** pega puntos `TAG,Q,H` (Q en m³/h o m³/s). La app ajusta `H0` y `K` y usa un duty `Q_ref,H_ref,eta,SG` para construir `T_load(n)`.

## Notas
- El modelo de sistema sustituye el `T_load ∝ n²` cuando hay puntos disponibles.
- Puedes calcular `r` desde `DriveRSheave` y `DriveNSheave` si cargas esas columnas.
