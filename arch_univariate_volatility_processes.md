# Univariate volatility processes in `arch` (Python)

## Notation

- Observed series: $y_t$
- Conditional mean (chosen separately in `arch`): $\mu_t$
- Innovation (residual): $\epsilon_t = y_t - \mu_t$
- Conditional variance: $\sigma_t^2 = \mathrm{Var}(\epsilon_t \mid \mathcal{F}_{t-1})$
- Often: $\epsilon_t = \sigma_t z_t$, where $z_t$ has $\mathbb{E}[z_t]=0$ and $\mathbb{E}[z_t^2]=1$
- Indicator: $I[A]$ equals 1 if event $A$ holds and 0 otherwise
- Lag operator: $L x_t = x_{t-1}$

---

## Constant variance (`ConstantVariance`)

$$
\sigma_t^2 = \sigma^2 \qquad (\sigma^2>0)
$$

---

## ARCH(p) (`ARCH`)

$$
\sigma_{t}^{2}=\omega+\sum_{i=1}^{p}\alpha_{i}\epsilon_{t-i}^{2}
$$

---

## Power GARCH and related models (`GARCH`)

`arch` uses a **power variance** specification that nests many common models:

```math
\sigma_t^{\lambda} = \omega
+ \sum_{i=1}^{p} \alpha_i \left|\epsilon_{t-i}\right|^{\lambda}
+ \sum_{j=1}^{o} \gamma_j \left|\epsilon_{t-j}\right|^{\lambda} I\left[\epsilon_{t-j}<0\right]
+ \sum_{k=1}^{q} \beta_k \sigma_{t-k}^{\lambda}
```

where $\lambda$ is the `power` argument.

Special cases:

- **ARCH(p)**: $q=0$, $o=0$, $\lambda=2$
- **GARCH(p,q)**: $o=0$, $\lambda=2$ (then $\sigma_t^\lambda=\sigma_t^2$)
- **GJR-GARCH(p,o,q)**: $o>0$, $\lambda=2$
- **AVARCH(p)**: $q=0$, $o=0$, $\lambda=1$
- **TARCH(p,o,q)**: $o>0$, $\lambda=1$
- **(A)VGARCH**: $\lambda=1$ with $q>0$

---

## EGARCH(p,o,q) (`EGARCH`)

Define standardized residuals $e_t = \epsilon_t / \sigma_t$. The variance dynamics are

$$
\ln\sigma_{t}^{2}=\omega
+\sum_{i=1}^{p}\alpha_{i}
\left(\left|e_{t-i}\right|-\sqrt{2/\pi}\right)
+\sum_{j=1}^{o}\gamma_{j} e_{t-j}
+\sum_{k=1}^{q}\beta_{k}\ln\sigma_{t-k}^{2}
$$

---

## FIGARCH(p,d,q) (`FIGARCH`)

In `arch` this class supports $p,q \in \{0,1\}$ and a power parameter (default `power=2.0`). One representation used in the docs is

$$
h_t = \omega + \left[1-\beta L - \phi L (1-L)^d\right]\epsilon_t^2 + \beta h_{t-1},
$$

where $L$ is the lag operator and $d$ is the fractional differencing parameter.

Estimation uses an ARCH($\infty$) representation

$$
h_t = (1-\beta)^{-1}\omega + \sum_{i=1}^{\infty}\lambda_i \epsilon_{t-i}^2,
$$

with weights built from

$$
\delta_1 = d,\qquad \lambda_1 = d-\beta+\phi,
$$

and the recursion (for $j\ge 2$)

$$
\delta_j = \frac{j-1-d}{j}\,\delta_{j-1},\qquad
\lambda_j = \beta \lambda_{j-1} + \delta_j - \phi \delta_{j-1}.
$$

When `power` is not 2, the same ARCH($\infty$) representation is used, replacing $\epsilon_t^2$ by $|\epsilon_t|^{p}$, where $p$ is the `power`.

---

## APARCH(p,o,q) (`APARCH`)

$$
\sigma_{t}^{\delta}=\omega
+\sum_{i=1}^{p}\alpha_{i}
\left(\left|\epsilon_{t-i}\right|
-\gamma_{i}I_{[o\geq i]}\epsilon_{t-i}\right)^{\delta}
+\sum_{k=1}^{q}\beta_{k}\sigma_{t-k}^{\delta}
$$

If `common_asym=True`, then all $\gamma_i$ are restricted to have a common value.

---

## HARCH (Heterogeneous ARCH) (`HARCH`)

$$
\sigma_{t}^{2}=\omega + \sum_{i=1}^{m}\alpha_{l_{i}}
\left(l_{i}^{-1}\sum_{j=1}^{l_{i}}\epsilon_{t-j}^{2}\right)
$$

where the model is specified by a set of lag-block lengths $\{l_i\}_{i=1}^m$.

Example with `lags=[1,5,22]`:

$$
\sigma_{t}^{2}=\omega+\alpha_{1}\epsilon_{t-1}^{2}
+\alpha_{5} \left(\frac{1}{5}\sum_{j=1}^{5}\epsilon_{t-j}^{2}\right)
+\alpha_{22} \left(\frac{1}{22}\sum_{j=1}^{22}\epsilon_{t-j}^{2}\right)
$$

---

## MIDAS Hyperbolic ARCH (`MIDASHyperbolic`)

$$
\sigma_{t}^{2}=\omega+
\sum_{i=1}^{m}\left(\alpha+\gamma I\left[\epsilon_{t-i}<0\right]\right)
\phi_{i}(\theta)\epsilon_{t-i}^{2}
$$

with hyperbolic weights

$$
\phi_{i}(\theta) \propto \frac{\Gamma(i+\theta)}{\Gamma(i+1)\Gamma(\theta)},
\qquad \sum_{i=1}^m \phi_i(\theta)=1.
$$

---

## EWMA / RiskMetrics (`EWMAVariance`)

$$
\sigma_t^{2}=\lambda\sigma_{t-1}^2 + (1-\lambda)\epsilon^2_{t-1},
\qquad 0<\lambda<1
$$

---

## RiskMetrics 2006 (`RiskMetrics2006`)

This is implemented as a **weighted average of multiple EWMA components**.

Define component time scales (cycles)

$$
\tau_k = \tau_1 \rho^{k-1}, \qquad k=1,\ldots,k_{\max}
$$

and smoothing parameters

$$
\mu_k = \exp\left(-\frac{1}{\tau_k}\right).
$$

The combination weights are constructed as

$$
w_k \propto 1 - \frac{\log(\tau_k)}{\log(\tau_0)},
\qquad \sum_{k=1}^{k_{\max}} w_k = 1.
$$

Each component variance follows an EWMA recursion

$$
h_{k,t} = \mu_k h_{k,t-1} + (1-\mu_k)\epsilon_{t-1}^2,
$$

and the reported variance is the weighted combination

$$
\sigma_t^2 = \sum_{k=1}^{k_{\max}} w_k h_{k,t}.
$$

---

## Fixed variance path (`FixedVariance`)

Given a user-supplied variance series $v_t$:

- If `unit_scale=True`: $\sigma_t^2 = v_t$.
- If `unit_scale=False` (default), a scale parameter $s>0$ is estimated and $\sigma_t^2 = s\,v_t$.

---

## `arch` references

- Docs: https://arch.readthedocs.io/en/stable/univariate/volatility.html
- Source: https://arch.readthedocs.io/en/latest/_modules/arch/univariate/volatility.html
