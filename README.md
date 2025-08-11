
# Memoria de Cálculo – Tiempos de Reacción de Bombas con VDF (MantoVerde)

**Versión:** v3 — densidad de pulpa, curva de sistema e inercias de transmisión  
**Objetivo:** estimar, por **TAG**, el tiempo para pasar de `n_i` a `n_f` al cambiar la consigna del VDF, considerando rampa del variador, par disponible e **inercia equivalente** del conjunto motor–transmisión–bomba–fluido.

---

## 1) Notación y unidades

| Símbolo | Descripción | Unidades |
|---|---|---|
| \(Q\) | Caudal | m³/s (usar m³/h ÷ 3600) |
| \(H\) | Altura manométrica total | m |
| \(\eta\) | Eficiencia hidráulica (eje–bomba) | – |
| \(\rho\) | Densidad del fluido (pulpa) \(\approx 1000 \cdot SG\) | kg/m³ |
| \(n\) | Velocidad (motor salvo se indique “bomba”) | rpm |
| \(\omega\) | Velocidad angular \(= 2\pi n/60\) | rad/s |
| \(r\) | Relación transmisión \(= n_\mathrm{motor}/n_\mathrm{bomba}\) | – |
| \(J\) | Momento de inercia | kg·m² |
| \(T\) | Par (torque) | N·m |

> GitHub **sí** renderiza LaTeX con $$ … $$ para bloques y $ … $ para fórmulas inline.

---

## 2) Par de carga con densidad de pulpa

Potencia hidráulica y par en el eje:
$$
P_h \;=\; \frac{\rho\,g\,Q\,H}{\eta}, 
\qquad
T_{\mathrm{load}} \;=\; \frac{P_h}{\omega}.
$$

Curva de sistema (régimen turbulento):
$$
H(Q) \;=\; H_0 \;+\; K\,Q^2.
$$

Relación caudal–velocidad cerca del duty de referencia:
$$
Q(n) \;\approx\; \alpha\,n, 
\qquad
\alpha \;=\; \frac{Q_\mathrm{ref}/3600}{n_\mathrm{ref}} \;\; [\mathrm{m^3/s~por~rpm}].
$$

**Efecto de densidad:** \(T_{\mathrm{load}} \propto \rho\). A mayor \(SG\), mayor \(T_{\mathrm{load}}\) y mayor tiempo de reacción.

---

## 3) Inercia equivalente reflejada al eje del motor

$$
J_{\mathrm{eq}} \;=\; J_m \;+\; J_{\mathrm{driver}} \;+\; r^2\!\left(J_{\mathrm{imp}} + J_{\mathrm{driven}} + J_{\mathrm{fluido}}\right).
$$

Límites/estimaciones geométricas:
$$
J_{\mathrm{imp,disco}} = \tfrac{1}{2} m R^2,
\qquad
J_{\mathrm{imp,aro}} = m R^2.
$$

Poleas (si se conoce el **peso de catálogo** sin bushing): \(J \approx m R^2\).  
Correas: masa equivalente como aro a radio de la polea.  
Fluido (opcional): \(J_{\mathrm{fluido}} \approx k\,m_{\mathrm{fluido}}\,R^2\), con \(m_{\mathrm{fluido}} \approx \rho\,\pi R^2 b\).

---

## 4) Rampa del VDF y tiempo por rampa

Rampa en la bomba limitada por la transmisión:
$$
\dot n_{\mathrm{bomba}} \;=\; \frac{\dot n_{\mathrm{motor}}}{r}.
$$

Tiempo mínimo puramente por rampa (sin carga):
$$
t_{\mathrm{rampa}} \;=\; \frac{n_{\mathrm{pump,max}} - n_{\mathrm{pump,min}}}{\dot n_{\mathrm{motor}}/r}.
$$

---

## 5) Dinámica por par/inercia

Ecuación de movimiento (lado motor):
$$
\frac{d\omega}{dt} \;=\; \frac{T_{\mathrm{avail}}(\omega) - T_{\mathrm{load}}(\omega)}{J_{\mathrm{eq}}}.
$$

Aceleración equivalente en rpm/s:
$$
\alpha_{\mathrm{torque}}(n) \;=\; \frac{60}{2\pi}\,\frac{T_{\mathrm{avail}} - T_{\mathrm{load}}(n)}{J_{\mathrm{eq}}}.
$$

En cada paso se impone el límite del VDF:
$$
\alpha(n) \;=\; \min\!\left\{\,\alpha_{\mathrm{torque}}(n),\ \dot n_{\mathrm{motor}}\,\right\},
\qquad
t \;=\; \int_{n_i}^{n_f} \frac{dn}{\alpha(n)}.
$$

**Tiempo final:**
$$
t_{\mathrm{final}} \;=\; \max\!\left\{\,t_{\mathrm{rampa}},\ t_{\mathrm{par}}\,\right\}.
$$

> Para **deceleración**, si no hay freno dinámico, el bus DC puede limitar la rampa efectiva y aumentar los tiempos.

---

## 6) Procedimiento resumido

1. Derivar \(r\) desde poleas: \(r \approx \mathrm{OD_{driven}}/\mathrm{OD_{driver}}\) (ajustar por *slip* si aplica).  
2. Calcular \(J_{\mathrm{eq}}\) (motor + driver + \(r^2\)(impulsor + driven + fluido/opcional)).  
3. Construir \(T_{\mathrm{load}}(n)\) con \(H_0,K,SG,\eta\) y \(Q_\mathrm{ref}\).  
4. Integrar de \(n_i\) a \(n_f\) con \(\alpha(n)=\min(\alpha_{\mathrm{torque}},\dot n_{\mathrm{motor}})\).  
5. Reportar \(t_{\mathrm{rampa}}\), \(t_{\mathrm{par}}\) y \(t_{\mathrm{final}}\).

---

## 7) Sensibilidades clave

- \(SG\) ↑ ⇒ \(T_{\mathrm{load}}\) ↑ ⇒ tiempos ↑ (aprox. lineal).  
- \(r\) ↑ ⇒ rampa efectiva de bomba ↓ y \(J\) reflejado ↑ (\(r^2\)).  
- Ø/masa de impulsor ↑ ⇒ \(J_{\mathrm{imp}}\) ↑ ⇒ tiempos ↑.  
- Poleas/bandas grandes (8V) ⇒ \(J\) adicional notable.  
- \(K\) alto (pérdidas cuadráticas) penaliza fuerte a altas rpm.  
- \(\eta\) ↓ ⇒ potencia y par requeridos ↑.

---

## 8) Campos del dataset (trazabilidad)

`TAG, SG, FrothFactor, Viscosidad_Pa_s, Q_ref_m3h, H_ref_m, Eta_ref, H0_m, K_m_per_m3s2, driver_series, driver_od_in, driver_grooves, driver_bushing, driver_shaft_mm, driven_series, driven_od_in, driven_grooves, driven_bushing, driven_shaft_mm, Jm_kgm2, Ø_imp_mm, M_imp_kg, n_motor_min, n_motor_max, n_pump_min, n_pump_max`

---

## 9) Ejecución rápida (app v3)

```bash
pip install -r requirements.txt
streamlit run app.py
```
