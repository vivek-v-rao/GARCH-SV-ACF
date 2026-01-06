### Analytic formulas (kurtosis + dependence of squared/absolute returns) for GARCH and SV processes

Code that compares Monte Carlo estimates to *analytic* (or analytic + controlled approximation) formulas for:

- **kurtosis** of returns `eps_t`
- **ACF / autocovariances** of **squared returns** `eps_t^2`
- **ACF / autocovariances** of **absolute returns** `|eps_t|`

for various GARCH models and the log autoregressive stochastic volatility (ARSV) model. The formulas below match what is implemented in `garch_acf.py` / `xgarch_acf.py`. Simulation results are [here](results.md). One reason to find expressions for time series properties in terms of GARCH parameters is to get an initial guess for the parameters used by an optimizer to maximum the log likelihood. Another is to compare the statistics implied by a volatility model fit to returns to the empirical statistics.

---

#### 1) Notation

Return process:
- `eps_t = sqrt(h_t) * z_t`
- `h_t` is the conditional variance
- `z_t` are iid innovations with:
  - `E[z_t] = 0`
  - `E[z_t^2] = 1`
  - `kappa = E[z_t^4]` (for standard normal, `kappa = 3`)
  - `ez_abs = E|z_t|` (for standard normal, `ez_abs = sqrt(2/pi)`)

Define:
- `mu_h = E[h_t]`
- `m2_h = E[h_t^2]`
- `var_h = Var(h_t) = m2_h - mu_h^2`

All ACFs are for **mean-corrected** series (i.e., based on covariances).

---

#### 2) Generic identities for squared returns

Because `z_t` is independent of `h_t` and independent over time:

- Mean of squared returns:
  - `E[eps_t^2] = E[h_t] = mu_h`

- Fourth moment of returns:
  - `E[eps_t^4] = E[h_t^2] * E[z_t^4] = kappa * m2_h`

- Variance of squared returns:
  - `Var(eps_t^2) = kappa * m2_h - mu_h^2`

- Kurtosis of returns:
  - `kurt(eps) = E[eps_t^4] / (E[eps_t^2])^2 = (kappa * m2_h) / mu_h^2`

- Autocovariance of squared returns (lag `k >= 1`):
  - `Cov(eps_t^2, eps_{t+k}^2) = E[h_t h_{t+k}] - mu_h^2`

Then:
- `acf_sq(k) = Cov(eps_t^2, eps_{t+k}^2) / Var(eps_t^2)`.

So the problem reduces to getting `mu_h`, `m2_h`, and `E[h_t h_{t+k}]`.

---

#### 3) Random-coefficient AR(1) form for h_t (Tier-A for squared returns)

Several models in the code can be written as:

`h_t = omega + A_{t-1} * h_{t-1}`

where `A_{t-1}` depends only on `z_{t-1}` (or its sign) and is iid over time.
Define the key expectations:

- `p1 = E[A]`
- `p2 = E[A^2]`
- `q1 = E[A * z^2]`

(Here `z` is a generic copy of `z_t`.)

#### 3.1 Unconditional moments of h_t

Under `p1 < 1` and `p2 < 1`:

- `mu_h = E[h] = omega / (1 - p1)`
- `m2_h = E[h^2] = omega^2 * (1 + p1) / ((1 - p1) * (1 - p2))`

#### 3.2 Autocovariances of squared returns eps^2

Define the lag-1 covariance of squared returns:

`cov1_sq = Cov(eps_t^2, eps_{t+1}^2) = omega * mu_h + q1 * m2_h - mu_h^2`

Then for `k >= 1`:

- `Cov(eps_t^2, eps_{t+k}^2) = p1^(k-1) * cov1_sq`

And (from section 2):

- `Var(eps^2) = kappa * m2_h - mu_h^2`
- `kurt(eps) = (kappa * m2_h) / mu_h^2`
- `acf_sq(k) = Cov(eps^2_t, eps^2_{t+k}) / Var(eps^2)`.

This is exactly what the generic `autocov_sq_from_pq(...)` path implements.

---

#### 4) Model-specific p1, p2, q1 for the random-coefficient class

#### 4.1 Symmetric GARCH(1,1)

Model:
- `h_t = omega + alpha * eps_{t-1}^2 + beta * h_{t-1}`
- with `eps_{t-1}^2 = h_{t-1} * z_{t-1}^2`, so
- `A = beta + alpha * z^2`

Thus:
- `p1 = alpha + beta`
- `p2 = beta^2 + 2*alpha*beta + alpha^2 * kappa`
- `q1 = beta + alpha * kappa`

#### 4.2 NAGARCH(1,1) (a.k.a. NGARCH, "2b" form in the code)

Model in random-coefficient form:
- `h_t = omega + [ beta + alpha*(z_{t-1} - theta)^2 ] * h_{t-1}`

Thus with general `kappa = E[z^4]`:
- `p1 = beta + alpha * (1 + theta^2)`
- `p2 = beta^2 + 2*alpha*beta*(1 + theta^2) + alpha^2 * (kappa + 6*theta^2 + theta^4)`
- `q1 = beta + alpha * (kappa + theta^2)`

(For normal `z`, `kappa = 3`.)

#### 4.3 GJR-GARCH(1,1)

Model:
- `h_t = omega + (alpha + gamma*I(z_{t-1}<0)) * eps_{t-1}^2 + beta*h_{t-1}`
- which becomes
- `h_t = omega + [ beta + (alpha + gamma*I(z<0)) * z^2 ] * h_{t-1}`

Assuming symmetric `z` so `P(z<0)=1/2` and `E[z^m I(z<0)] = 0.5*E[z^m]` for even `m`:

- `p1 = beta + alpha + 0.5*gamma`
- `p2 = beta^2 + 2*beta*(alpha + 0.5*gamma) + kappa*(alpha^2 + alpha*gamma + 0.5*gamma^2)`
- `q1 = beta + kappa*(alpha + 0.5*gamma)`

---

#### 5) Closed-form ACF of squared returns for GARCH(1,1) via ARMA(1,1)

The code also uses the classic ARMA(1,1) representation for `eps_t^2`:

`eps_t^2 = const + phi*eps_{t-1}^2 + u_t + theta*u_{t-1}`

with:
- `phi = alpha + beta`
- `theta = -beta`

For an ARMA(1,1) process, the ACF is:

- `rho_sq(k) = phi^(k-1) * rho_sq(1)` for `k >= 1`

where:

`rho_sq(1) = (phi + theta) * (1 + phi*theta) / (1 + 2*phi*theta + theta^2)`.

This is exactly the formula implemented by `theoretical_acf_sq_garch_1_1(...)`.

---

#### 6) ACF of squared returns for NAGARCH and GJR-GARCH (normal z)

For NAGARCH(1,1) and GJR-GARCH(1,1), the code uses:

- `rho_sq(k) = p1^(k-1) * rho_sq(1)` for `k >= 1`

with:

`rho_sq(1) = ((1 - p1^2)*q1 - p1*(1 - p2)) / (2 + p2 - 3*p1^2)`.

(Here `p1, p2, q1` are as in section 4.2 or 4.3.)

---

#### 7) Absolute returns |eps_t| : Tier-A lognormal approximation (GARCH / NAGARCH / GJR / FIGARCH)

The code's "Tier-A" approximation treats `h_t` as **lognormal** by matching
its mean and variance:

Assume:
- `h_t = exp(U_t)`
- `U_t ~ N(m, s2)` marginally

Match the first two moments:

- `s2 = log( 1 + var_h / mu_h^2 )`
- `m  = log(mu_h) - 0.5*s2`

#### 7.1 Converting Corr(h_t, h_{t+k}) to Corr(U_t, U_{t+k})

Let:
- `corr_h(k) = Corr(h_t, h_{t+k})`.

Then for the corresponding Gaussian correlation `rho_u(k)`:

- `rho_u(k) = log( 1 + corr_h(k) * (exp(s2) - 1) ) / s2`.

#### 7.2 Moments needed for |eps|

For any `p`:
- `E[h^p] = exp(p*m + 0.5*p^2*s2)`

So:
- `E[sqrt(h)] = E[h^(1/2)] = exp(0.5*m + 0.125*s2)`

Then:
- `mean_abs = E|eps| = ez_abs * E[sqrt(h)]`

And since `|eps|^2 = eps^2` and `E[z^2]=1`:
- `Var(|eps|) = E[eps^2] - mean_abs^2 = mu_h - mean_abs^2`

#### 7.3 Cross-moment and autocovariance of |eps|

Assume `(U_t, U_{t+k})` is bivariate normal with correlation `rho_u(k)`.
For constants `a,b`:

`E[ exp(a U_t + b U_{t+k}) ] = exp( (a+b)*m + 0.5*(a^2 + b^2 + 2ab*rho_u(k))*s2 )`.

With `a=b=1/2`:

- `E[ sqrt(h_t) * sqrt(h_{t+k}) ] = exp( m + 0.25*(1 + rho_u(k))*s2 )`

Tier-A then approximates:
- `E[ |eps_t| |eps_{t+k}| ] ~= ez_abs^2 * E[ sqrt(h_t) sqrt(h_{t+k}) ]`

So:
- `Cov(|eps_t|, |eps_{t+k}|) ~= ez_abs^2 * ( exp(m + 0.25*(1+rho_u(k))*s2) - (E[sqrt(h)])^2 )`

Finally:
- `acf_abs(k) = Cov(|eps_t|, |eps_{t+k}|) / Var(|eps|)`.

#### 7.4 Corr(h_t, h_{t+k}) used in Tier-A

For the random-coefficient AR(1) class (sections 3-4), the code uses:

- `Cov(h_t, h_{t+k}) = p1^k * var_h`
- hence `corr_h(k) = p1^k`.

For FIGARCH, `corr_h(k)` is obtained via the ARFIMA-based autocovariances
(see sections 8.2 and 8.3).

---

#### 8) FIGARCH(1,d,1)

#### 8.1 Weights lambda_j for the ARCH(infinity) representation

The code builds the fractional-difference coefficients `pi_j` by:

- `pi_0 = 1`
- `pi_j = pi_{j-1} * ( (j-1-d) / j )`, `j >= 1`

With `phi = alpha + beta`, the ARCH(infinity) weights are:

- `lambda_j = pi_{j+1} - phi * pi_j`, for `j >= 0`.

(These are computed up to a truncation level `n_terms`.)

#### 8.2 ACF of squared returns eps^2 via ARFIMA(1,d,1)

Let:
- `x_t = eps_t^2 - E[eps_t^2]`.

The code uses the ARFIMA representation:

`(1 - ar1*L) (1 - L)^d x_t = (1 - ma1*L) nu_t`

with:
- `ar1 = alpha + beta`
- `ma1 = beta`
- `nu_t = h_t (z_t^2 - 1)` with `Var(nu_t) = sigma2`

Compute:
- `pi_0 = 1`, `pi_j = pi_{j-1} * ((j-1+d)/j)`  (note the sign convention here matches the code's ARFIMA expansion)
- `b_0 = pi_0`
- for `j>=1`: `b_j = pi_j + ma1*pi_{j-1}`
- `psi_0 = b_0`
- for `j>=1`: `psi_j = b_j + ar1*psi_{j-1}`

Then the autocovariance is:

- `gamma_x(k) = sigma2 * sum_{j>=0} psi_j * psi_{j+k}`

and:
- `acf_sq(k) = gamma_x(k) / gamma_x(0)`.

#### 8.3 Tier-A absolute returns for FIGARCH

The code first computes `mu_h` from the ARCH(infinity) weights:

- `const = omega / (1 - beta)`
- `s1 = sum_j lambda_j`
- `mu_h = const / (1 - s1)`

Let `base_gamma_x(k)` be the ARFIMA autocovariances from 8.2 computed with `sigma2 = 1`,
and let `base0 = base_gamma_x(0)`.

Then it solves for `m2_h = E[h^2]` by matching the identity:

- `Var(eps^2) = (kappa - 1)*m2_h * base0 = kappa*m2_h - mu_h^2`

which gives:

- `m2_h = mu_h^2 / ( kappa - (kappa - 1)*base0 )`

Then:
- `var_h = m2_h - mu_h^2`
- `sigma2 = (kappa - 1)*m2_h`
- `Cov(h_t, h_{t+k}) = sigma2 * base_gamma_x(k)`

From there, Tier-A proceeds as in section 7.

---

#### 9) EGARCH(1,1) with normal innovations (exact up to truncation)

Simulation model (Nelson-style):
- `eps_t = sqrt(h_t) z_t`, `z_t ~ N(0,1)`
- `x_t = log(h_t)`
- `x_t = omega + beta*x_{t-1} + alpha*(|z_{t-1}| - E|z|) + gamma*z_{t-1}`

Write the stationary solution:

- `x = omega/(1-beta) + sum_{j>=0} beta^j u_{-1-j}`
- `u = alpha(|z| - E|z|) + gamma z`

Define the normal-specific function:

- `M_u(t) = E[ exp( t * u ) ]`

Then for any scalar `s`:

- `E[exp(s x)] = exp( s*omega/(1-beta) ) * prod_{j>=0} M_u( s*beta^j )`

The code truncates the product when `|s*beta^j|` becomes small.

#### 9.1 Closed forms used for M_u(t), Q_u(t), A_u(t)

For `z ~ N(0,1)` and `ez_abs = E|z| = sqrt(2/pi)`:

Let `a = t*alpha`, `g = t*gamma`, and `Phi(.)` be the standard normal CDF.

- `M_u(t) = exp(-a*ez_abs) * [ exp((a+g)^2/2)*Phi(a+g) + exp((g-a)^2/2)*Phi(a-g) ]`

The code also uses:
- `Q_u(t) = E[ z^2 * exp( t*u ) ]`
- `A_u(t) = E[ |z| * exp( t*u ) ]`

via auxiliary half-line integrals (also in closed form).

#### 9.2 Squared returns: autocov and kurtosis

The EGARCH routines compute (up to truncation):
- `mu_h = E[h] = E[exp(x)]`
- `m2_h = E[h^2] = E[exp(2x)]`
- `E[h_t h_{t+k}]` via the same product method

Then they plug into section 2 identities:
- `Var(eps^2) = kappa*m2_h - mu_h^2`
- `kurt(eps) = (kappa*m2_h)/mu_h^2`
- `Cov(eps_t^2, eps_{t+k}^2) = E[h_t h_{t+k}] - mu_h^2`
- `acf_sq(k) = Cov / Var`.

#### 9.3 Absolute returns: exact autocov (up to truncation)

The EGARCH absolute-return autocovariance uses the exact decomposition:

`E[ |eps_t| |eps_{t+k}| ] = E[ |z_t| * exp(0.5*x_t) * |z_{t+k}| * exp(0.5*x_{t+k}) ]`

and evaluates it using:
- the product formulas for `E[exp(s x)]` with `s = 0.5*(1+beta^k)`
- the closed-form `A_u(.)` term for the single innovation that links `x_t` and `x_{t+k}`

Then:
- `mean_abs = E|eps| = ez_abs * E[exp(0.5 x)]`
- `Var(|eps|) = E[h] - mean_abs^2`
- `Cov(|eps_t|,|eps_{t+k}|) = E[|eps_t||eps_{t+k}|] - mean_abs^2`
- `acf_abs(k) = Cov / Var`.

(These are implemented by `autocov_abs_egarch_1_1(...)` and `acf_abs_egarch_1_1(...)`.)

---

#### 10) Stochastic volatility: log-AR(1) (squared returns only in the code)

One SV form in the code is:

- `x_t = log(h_t)`
- `x_t = mu + phi*(x_{t-1} - mu) + sigma_eta * eta_t`, `eta_t ~ N(0,1)`
- so `h_t = exp(x_t)` is lognormal, and `(x_t, x_{t+k})` is bivariate normal.

Let:
- `Var(x) = sigma_eta^2 / (1 - phi^2)`
- `Cov(x_t, x_{t+k}) = Var(x) * phi^k`

Then:
- `mu_h = E[h] = exp( mu + 0.5*Var(x) )`
- `m2_h = E[h^2] = exp( 2*mu + 2*Var(x) )`
- `E[h_t h_{t+k}] = exp( 2*mu + Var(x) + Cov(x_t, x_{t+k}) )`

Plug into section 2:
- `Var(eps^2) = kappa*m2_h - mu_h^2`
- `Cov(eps_t^2, eps_{t+k}^2) = E[h_t h_{t+k}] - mu_h^2`
- `kurt(eps) = (kappa*m2_h)/mu_h^2`
- `acf_sq(k) = Cov/Var`.

(Absolute-return ACF for SV is not currently provided in the codebase.)

---

#### 11) Stationarity / finiteness conditions used implicitly

- Random-coefficient class (GARCH/NAGARCH/GJR):
  - `p1 < 1` for finite mean
  - `p2 < 1` for finite second moment of `h` (and hence finite `Var(eps^2)`)

- EGARCH:
  - `|beta| < 1` for covariance stationarity of `x = log(h)`

- FIGARCH:
  - typically `0 < d < 1`, plus truncation is used for numerical evaluation.

---

#### 12) References

- GARCH moment / ACF formulas agree with He & Terasvirta, [Properties of moments of a family of GARCH processes](papers/properties_of_moments_of_a_family_of_garch_processes.pdf).
- EGARCH moment / ACF formulas implemented here follow the "moments of ARMA-EGARCH" approach:
  - Karanasos & Kim, [Moments of the ARMA-EGARCH Model](papers/moments_of_the_arma_egarch_model.pdf).
- FIGARCH formulas agree with Karanasos et al., [On the Autocorrelation Properties of Long‐Memory GARCH Processes](papers/autocorrelation_properties_of_long_memory_garch_processes.pdf).
