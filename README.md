
# Memoria de Cálculo – Tiempos de Reacción de Bombas con VDF (MantoVerde)

**Versión:** v3 — densidad de pulpa, curva de sistema e inercias de transmisión  
**Alcance:** Estimar el **tiempo de reacción** (aceleración/deceleración) de cada bomba ante cambios de consigna del VDF, considerando límites de rampa del variador, par disponible del motor y **dinámica por par/inercia** del tren de accionamiento y del fluido.

---

## 1. Objetivo y contexto

Determinar, por **TAG**, el tiempo requerido para pasar de una velocidad inicial `n_i` a una velocidad final `n_f` de bomba cuando el VDF cambia consigna. El modelo incorpora:
- **Rampa del VDF** (límite de cambio de velocidad en el **motor**, rpm/s);
- **Curva de sistema** para obtener el **par de carga** en función del caudal/velocidad y la **densidad de pulpa**;
- **Inercias equivalentes** reflejadas al eje del motor: rotor, poleas (driver/driven), correas (masa equivalente), impulsor y fluido.

Se reportan: `t_ramp_only` (límite por VDF), `t_par` (límite por par/inercia) y `t_final = max(t_ramp_only, t_par)`.

---

## 2. Datos de entrada (por TAG)

| Categoría | Variables (unidades) | Descripción |
|---|---|---|
| Motor | `T_nom` (N·m), `J_m` (kg·m²), `n_motor_min/max` (rpm) | Par nominal a velocidad nominal; momento de inercia del rotor; velocidades de operación. |
| Transmisión | `driver_series` (5V/8V), `driver_od_in` (in), `driver_grooves` (-), `driver_bushing` (serie), `driver_shaft_mm` (mm); `driven_*` análogo; **distancia entre centros** (opcional) | Para derivar **r = n_motor / n_bomba ≈ OD_driven/OD_driver** y estimar **inercias**. |
| Bomba/Impulsor | `Ø_imp` (mm), `M_imp` (kg), `n_pump_min/max` (rpm) | Para `J_imp` y ventana de operación. |
| Pulpa y sistema | `SG` (-), `FrothFactor` (-), `Viscosidad` (Pa·s); `Q_ref` (m³/h), `H_ref` (m), `η_ref` (-); `H0` (m), `K` (m/(m³/s)²) | Densidad relativa, espuma, viscosidad; **duty** de referencia y **parámetros de la curva de sistema** `H = H0 + K·Q²`. |

> El `initial_dataset.csv` contiene **SG/FF/μ**, `Q_ref/H_ref/η_ref` y `H0/K` precargados por TAG. La tabla de poleas `sheaves_default.csv` trae series/diámetros/ranuras/bushings iniciales.

---

## 3. Supuestos principales

1. **Zona de par constante** del motor hasta su velocidad nominal (sin debilitamiento de campo).  
2. **Rampa del VDF** aplicada en el **eje del motor**: `ramp_motor` (rpm/s).  
3. **Relación de transmisión** `r = n_motor / n_bomba ≈ OD_driven / OD_driver · (1/(1−slip))`.  
4. **Curva de sistema** en el rango de interés:  
   \[ H(Q) \approx H_0 + K\,Q^2 \quad \text{con } Q\ \text{en m}^3/\text{s} \]  
5. **Afinidad bomba (screening)**: \(Q \propto n\), \(H \propto n^2\) (solo para relacionar \(Q\) y \(n\)).  
6. **Inercias** aproximadas como aros delgados donde aplique; correas como **masa equivalente** anular.  
7. Eficiencia hidráulica **η** constante alrededor del duty (ajustable por TAG).

---

## 4. Modelo matemático

### 4.1. Par de carga con densidad de pulpa
Potencia hidráulica y par en el eje de la bomba:
\[
P_h = \frac{\rho\,g\,Q\,H}{\eta}, \qquad
T_{\text{load,bomba}} = \frac{P_h}{\omega_{\text{bomba}}}
\]

Con \(Q(n) \approx \alpha\,n\), \(\alpha = Q_{\text{ref}}/n_{\text{ref}}\) (en m³/s por rpm) y \(H(Q)=H_0+KQ^2\), el **par equivalente en el eje del motor** (misma magnitud, evaluado a \(n\) del **motor**) se usa en la integración. La **densidad** \(\rho = 1000\cdot SG\) **afecta linealmente** al par de carga: a mayor SG, mayor \(T_{\text{load}}\) y **mayor tiempo**.

### 4.2. Inercia equivalente reflejada al eje del motor
\[
J_{\text{eq}} = J_m \;+\; J_{\text{driver}} \;+\; r^2\,(J_{\text{imp}}+J_{\text{driven}}+J_{\text{fluido}})
\]

- **Impulsor** (cotas geométricas):  
  \(\displaystyle J_{\text{imp,disco}} = \tfrac{1}{2} m R^2,\quad J_{\text{imp,aro}} = m R^2\)  
- **Poleas** (catálogo TB Wood’s): si hay **peso** \(m\), usar \(J \approx mR^2\) (aro); si el peso **incluye bushing**, restarlo con la tabla de bushings.  
- **Correas**: masa equivalente como aro a radio de la polea correspondiente (parámetro editable).  
- **Fluido** (opcional): \(J_{\text{fluido}} \approx k\,m_{\text{fluido}}\,R^2\), \(m_{\text{fluido}} \approx \rho\cdot \pi R^2 b\) (ancho hidráulico \(b\)).

### 4.3. Rampa del VDF
- En el **motor**: \(\dot{n}_{\max}^{(m)}=\text{ramp}_{\text{motor}}\) (rpm/s)  
- En la **bomba**: \(\dot{n}_{\max}^{(b)}=\text{ramp}_{\text{motor}}/r\) (rpm/s)

Tiempo mínimo puramente por rampa (ideal, sin carga):
\[
t_{\text{ramp,\,bomba}}=\frac{n_{\text{pump,max}}-n_{\text{pump,min}}}{\text{ramp}_{\text{motor}}/r}
\]

### 4.4. Dinámica por par/inercia (motor)
\[
\frac{d\omega}{dt}=\frac{T_{\text{avail}}(\omega)-T_{\text{load}}(\omega)}{J_{\text{eq}}}, \qquad
\alpha_{\text{torque}}=\frac{60}{2\pi}\frac{T_{\text{avail}}-T_{\text{load}}}{J_{\text{eq}}}\ (\text{rpm/s})
\]

En cada paso:
\[
\alpha(n)=\min\big(\alpha_{\text{torque}}(n),\ \text{ramp}_{\text{motor}}\big)
\quad\Rightarrow\quad
t \;+=\; \frac{\Delta n_{\text{motor}}}{\alpha(n)}.
\]

Tiempo por par/inercia \(t_{\text{par}}\) y **tiempo final**:
\[
t_{\text{final}}=\max\big(t_{\text{ramp,\,bomba}},\ t_{\text{par}}\big).
\]

> Para **deceleración** puede requerirse freno dinámico; si no, la rampa efectiva puede quedar limitada por el bus DC y el tiempo aumentar.

---

## 5. Procedimiento de cálculo (paso a paso)

1. **Entradas por TAG**: cargar/editar `initial_dataset.csv` (SG, FF, μ, duty, H0, K) y `sheaves_default.csv` (series/OD/ranuras/bushings).  
2. **Derivar `r`** desde poleas: \(r \approx \frac{OD_{\text{driven}}}{OD_{\text{driver}}}\cdot\frac{1}{1-\text{slip}}\).  
3. **Calcular inercias**: \(J_m\) (motor), \(J_{\text{driver}}\), \(J_{\text{driven}}\), \(J_{\text{imp}}\) (disco/aro), \(J_{\text{fluido}}\) (opcional).  
4. **Construir \(J_{\text{eq}}\)** reflejando elementos al eje motor.  
5. **Calcular \(T_{\text{load}}(n)\)** con `H0/K`, `SG`, `η` y `Q_ref`.  
6. **Integrar** de `n_motor_min` a `n_motor_max` (y viceversa si aplica), imponiendo **límite de rampa** del VDF.  
7. **Reportar** `t_ramp_only`, `t_par` y `t_final`.  
8. **Validar** corriente pico vs. límites del VDF y **NPSH** en los puntos acelerados.

---

## 6. Sensibilidades y efectos principales

- **Densidad (SG)**: \(T_{\text{load}}\propto \rho\) ⇒ tiempos crecen **casi linealmente** con SG.  
- **Relación de transmisión (r)**: a mayor `r`, menor rampa efectiva en la bomba \((\dot{n}_b=\dot{n}_m/r)\) y mayor \(J\) reflejado \((r^2)\).  
- **Impulsor (Ø, masa)**: aumenta \(J_{\text{imp}}\) y el tiempo por inercia.  
- **Poleas/correas**: añaden \(J\) (sobre todo en 8V grandes).  
- **Curva de sistema (H0/K)**: mayor pérdida cuadrática (K) sube mucho \(T_{\text{load}}\) a alto caudal → tiempos mayores.  
- **η**: menor eficiencia eleva \(P_h\) requerido → sube \(T_{\text{load}}\).  
- **Rampa del VDF**: fija el **piso** de tiempo; por debajo de cierta carga, manda la rampa.

---

## 7. Validaciones y chequeos recomendados

- **Corriente/pu** del variador vs. tiempo permitido (overload).  
- **NPSH disponible** durante la aceleración (no cavitar).  
- **Límites de operación** del impulsor y régimen de espuma (FF).  
- **Resonancias**/limitaciones mecánicas en regiones de paso rápido.  
- Coherencia de `H0/K` con los puntos **Base/Opt** y con la hidráulica de la línea.

---

## 8. Implementación en la app (resumen)

- **Entradas editables**: rampa VDF, `T_nom` por TAG (puede ser pu), tablas de **dataset** e **inercias/poleas**.  
- **Cálculo**: integración numérica de \(n_i \rightarrow n_f\) en 600–800 pasos, con \(\alpha(n)=\min(\alpha_{\text{torque}},\ \text{rampa})\).  
- **Salidas**: `t_ramp_only`, `t_par`, `t_final`, `J_eq` desglosado y parámetros usados (`SG, η, H0, K, Q_ref`).  
- **Descargas**: CSV de entradas y resultados.

---

## 9. Campos del dataset (para trazabilidad)

`TAG, SG, FrothFactor, Viscosidad_Pa_s, Q_ref_m3h, H_ref_m, Eta_ref, H0_m, K_m_per_m3s2, driver_series, driver_od_in, driver_grooves, driver_bushing, driver_shaft_mm, driven_series, driven_od_in, driven_grooves, driven_bushing, driven_shaft_mm, Jm_kgm2, Ø_imp_mm, M_imp_kg, n_motor_min, n_motor_max, n_pump_min, n_pump_max`

---

## 10. Limitaciones

- `H = H0 + K Q^2` es una **aproximación** (válida en régimen turbulento; no modela válvulas/singularidades no cuadráticas ni transitorios de línea).  
- `η` fijo alrededor de `Q_ref`; podría variarse con \(Q\).  
- Inercias de poleas/correas son aproximaciones si no se carga **peso de catálogo**.  
- No se modela el control de **nivel de cajones** ni retroalimentación del sistema durante el tránsito.

---

## 11. Próximas mejoras sugeridas

- Importar **automáticamente** `H0/K`, SG y puntos Q–H desde los **PDF** de dimensionamiento.  
- Agregar curvas **η(Q)** y límites de **corriente/tiempo** del VDF por TAG.  
- Modo **deceleración** con **freno dinámico** y rampa asimétrica.  
- Considerar **longitud de correa/centros** para evaluar rangos de ajuste y masa real de correas.

---

## 12. Resultados esperados (ejemplo de interpretación)

- Si `t_ramp_only << t_par`: la **dinámica por par/inercia** domina; conviene revisar **SG**, `H0/K`, `η` y **J** del tren.  
- Si `t_ramp_only >> t_par`: la **rampa del VDF** limita; para mejorar, subir rampa (si corriente/NPSH lo permiten) o reducir `r`.  
- `J_eq` alto por poleas 8V grandes y/o impulsor pesado ⇒ tiempos más largos incluso con motor sobredimensionado.

---

## 13. Cómo ejecutar la app (referencia)

```bash
pip install streamlit pandas numpy matplotlib
streamlit run streamlit_app_bombas_vdf_v3.py
```

> Cargar/editar `initial_dataset.csv` y `sheaves_default.csv`. Los resultados pueden exportarse a CSV y anexarse a esta memoria.
