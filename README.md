
# Memoria de Cálculo – Tiempos de Reacción de Bombas con VDF (MantoVerde)

**Versión:** v3 — densidad de pulpa, curva de sistema e inercias de transmisión

## 1) Notación y unidades

| Símbolo | Descripción | Unidades |
|---|---|---|
| \(Q\) | Caudal | m³/s (usar m³/h ÷ 3600) |
| \(H\) | Altura manométrica total | m |
| \(\eta\) | Eficiencia | – |
| \(\rho\) | Densidad de pulpa \(= 1000 \cdot SG\) | kg/m³ |
| \(n\) | Velocidad (motor salvo que se indique) | rpm |
| \(\omega\) | \(2\pi n/60\) | rad/s |
| \(r\) | Relación \(n_m/n_b\) | – |
| \(J\) | Inercia | kg·m² |
| \(T\) | Par | N·m |

## 2) Par de carga con densidad de pulpa

**Potencia hidráulica y par en el eje:**

$$
P_h=\frac{\rho\,g\,Q\,H}{\eta}, \qquad
T_{\mathrm{load}}=\frac{P_h}{\omega}.
$$

**Curva de sistema (régimen turbulento):**

$$
H(Q)=H_0+K\,Q^2.
$$

**Relación caudal–velocidad cerca del duty de referencia:**

$$
Q(n)\approx \alpha\,n, \qquad
\alpha = \frac{Q_{\mathrm{ref}}/3600}{n_{\mathrm{ref}}} \;\; [\mathrm{m^3\,s^{-1}\,por\,rpm}].
$$

**Efecto de densidad:** \(T_{\mathrm{load}}\propto \rho\). A mayor \(SG\), mayor \(T_{\mathrm{load}}\) y mayor tiempo de reacción.

## 3) Inercia equivalente reflejada al eje del motor

$$
J_{\mathrm{eq}} = J_m + J_{\mathrm{driver}} + r^2\!\left(J_{\mathrm{imp}} + J_{\mathrm{driven}} + J_{\mathrm{fluido}}\right).
$$
