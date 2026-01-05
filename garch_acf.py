# Warning!
#
# squared-return ACF for GARCH/NAGARCH/GJR and the crosscorr blocks look correct.
# 
# Not expected to match: the |eps| ACF/ACV "Tier A" blocks (systematically low is normal).
# 
# FIGARCH:
# 
# squared-return ACF mismatch is likely truncation-vs-infinite-theory + noise,
# 
# |eps| theory showing repeated identical early lags strongly suggests a real conceptual/input problem (corr(h) clipping) in the FIGARCH absolute-return mapping.
# 
# EGARCH squared-return ACF mismatch is consistent with your stated approximation being too rough.

import math
import numpy as np


def simulate_garch_1_1(n, omega, alpha, beta, burn=5000, seed=12345):
    """
    Simulate a symmetric GARCH(1,1):

        eps_t = sqrt(h_t) * z_t,  z_t ~ N(0,1)
        h_t = omega + alpha * eps_{t-1}^2 + beta * h_{t-1}

    Returns
    -------
    eps : ndarray, shape (n,)
    h   : ndarray, shape (n,)
    """
    if omega <= 0:
        raise ValueError("omega must be > 0")
    if alpha < 0 or beta < 0:
        raise ValueError("alpha and beta must be >= 0")
    if alpha + beta >= 1:
        raise ValueError("need alpha + beta < 1 for covariance stationarity")

    rng = np.random.default_rng(seed)
    total = n + burn

    eps = np.empty(total, dtype=float)
    h = np.empty(total, dtype=float)

    # start at unconditional variance
    h[0] = omega / (1.0 - alpha - beta)
    eps[0] = math.sqrt(h[0]) * rng.standard_normal()

    for t in range(1, total):
        h[t] = omega + alpha * (eps[t - 1] ** 2) + beta * h[t - 1]
        eps[t] = math.sqrt(h[t]) * rng.standard_normal()

    return eps[burn:], h[burn:]


def simulate_nagarch_1_1(n, omega, alpha, beta, theta, burn=5000, seed=12345):
    """
    Simulate NAGARCH/NGARCH in the "2b" form (normal innovations):

        eps_t = sqrt(h_t) * z_t,  z_t ~ N(0,1)
        h_t = omega + [ beta + alpha*(z_{t-1} - theta)^2 ] * h_{t-1}

    Equivalent to:
        h_t = omega + beta*h_{t-1} + alpha*(eps_{t-1} - theta*sqrt(h_{t-1}))^2

    Returns
    -------
    eps : ndarray, shape (n,)
    h   : ndarray, shape (n,)
    """
    if omega <= 0:
        raise ValueError("omega must be > 0")
    if alpha < 0 or beta < 0:
        raise ValueError("alpha and beta must be >= 0")

    p1 = beta + alpha * (1.0 + theta * theta)
    if p1 >= 1:
        raise ValueError("need beta + alpha*(1+theta^2) < 1 for E[h] to exist")

    rng = np.random.default_rng(seed)
    total = n + burn

    eps = np.empty(total, dtype=float)
    h = np.empty(total, dtype=float)

    # start at unconditional mean of h
    h[0] = omega / (1.0 - p1)
    eps[0] = math.sqrt(h[0]) * rng.standard_normal()

    for t in range(1, total):
        z_prev = eps[t - 1] / math.sqrt(h[t - 1])
        h[t] = omega + (beta + alpha * (z_prev - theta) ** 2) * h[t - 1]
        eps[t] = math.sqrt(h[t]) * rng.standard_normal()

    return eps[burn:], h[burn:]


def simulate_gjr_garch_1_1(n, omega, alpha, beta, gamma, burn=5000, seed=12345):
    """
    Simulate GJR-GARCH(1,1) with normal innovations:

        eps_t = sqrt(h_t) * z_t,  z_t ~ N(0,1)
        h_t = omega + alpha*eps_{t-1}^2 + gamma*eps_{t-1}^2*I_{t-1} + beta*h_{t-1}
        I_{t-1} = 1{eps_{t-1} < 0}

    Under normal innovations, E[I] = 1/2, so E[h] exists if
        beta + alpha + gamma/2 < 1.

    Returns
    -------
    eps : ndarray, shape (n,)
    h   : ndarray, shape (n,)
    """
    if omega <= 0:
        raise ValueError("omega must be > 0")
    if alpha < 0 or beta < 0 or gamma < 0:
        raise ValueError("alpha, beta, gamma must be >= 0")

    p1 = beta + alpha + 0.5 * gamma
    if p1 >= 1:
        raise ValueError("need beta + alpha + gamma/2 < 1 for E[h] to exist")

    rng = np.random.default_rng(seed)
    total = n + burn

    eps = np.empty(total, dtype=float)
    h = np.empty(total, dtype=float)

    # start at unconditional mean of h
    h[0] = omega / (1.0 - p1)
    eps[0] = math.sqrt(h[0]) * rng.standard_normal()

    for t in range(1, total):
        i_neg = 1.0 if eps[t - 1] < 0.0 else 0.0
        h[t] = omega + alpha * (eps[t - 1] ** 2) + gamma * (eps[t - 1] ** 2) * i_neg + beta * h[t - 1]
        eps[t] = math.sqrt(h[t]) * rng.standard_normal()

    return eps[burn:], h[burn:]


def acf(x, nlags):
    """
    Sample autocorrelation function for lags 0..nlags.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom == 0:
        raise ValueError("zero variance series")

    out = np.empty(nlags + 1, dtype=float)
    out[0] = 1.0
    for k in range(1, nlags + 1):
        out[k] = np.dot(x[k:], x[:-k]) / denom
    return out


def crosscorr_lead(x, y, nlags):
    """
    Sample cross-correlation for lags 0..nlags:

        out[k] = Corr(x_t, y_{t+k})

    Uses global de-meaning and a fixed normalization across lags.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same length")

    x = x - x.mean()
    y = y - y.mean()

    denom = math.sqrt(np.dot(x, x) * np.dot(y, y))
    if denom == 0:
        raise ValueError("zero variance in x or y")

    out = np.empty(nlags + 1, dtype=float)
    out[0] = np.dot(x, y) / denom
    for k in range(1, nlags + 1):
        out[k] = np.dot(x[:-k], y[k:]) / denom
    return out


def theoretical_acf_sq_garch_1_1(alpha, beta, nlags):
    """
    Theoretical ACF of eps_t^2 for symmetric GARCH(1,1) with iid innovations
    and the ARMA(1,1) mapping:

        eps_t^2 = const + phi * eps_{t-1}^2 + u_t + theta * u_{t-1}
        phi   = alpha + beta
        theta = -beta

    For ARMA(1,1): rho(k) = phi^(k-1) * rho(1), k>=1
    where rho(1) = (phi + theta) * (1 + phi*theta) / (1 + 2*phi*theta + theta^2).
    """
    phi = alpha + beta
    theta = -beta

    denom = 1.0 + 2.0 * phi * theta + theta * theta
    rho1 = (phi + theta) * (1.0 + phi * theta) / denom

    out = np.empty(nlags + 1, dtype=float)
    out[0] = 1.0
    for k in range(1, nlags + 1):
        out[k] = (phi ** (k - 1)) * rho1
    return out


def theoretical_acf_sq_nagarch_1_1(alpha, beta, theta, nlags):
    """
    Theoretical ACF of eps_t^2 for NAGARCH(1,1) "2b" with z_t ~ N(0,1):

        h_t = omega + [ beta + alpha*(z_{t-1}-theta)^2 ] h_{t-1}
        eps_t = sqrt(h_t) z_t

    rho(k) = p1^(k-1) * rho(1), k>=1
    """
    p1 = beta + alpha * (1.0 + theta * theta)
    p2 = (
        beta * beta
        + 2.0 * alpha * beta * (1.0 + theta * theta)
        + alpha * alpha * (3.0 + 6.0 * theta * theta + theta ** 4)
    )
    q1 = beta + alpha * (3.0 + theta * theta)

    if p1 >= 1.0:
        raise ValueError("need beta + alpha*(1+theta^2) < 1")
    if p2 >= 1.0:
        raise ValueError("need p2 < 1 (finite second moment of h)")

    denom = 2.0 + p2 - 3.0 * p1 * p1
    rho1 = ((1.0 - p1 * p1) * q1 - p1 * (1.0 - p2)) / denom

    out = np.empty(nlags + 1, dtype=float)
    out[0] = 1.0
    for k in range(1, nlags + 1):
        out[k] = (p1 ** (k - 1)) * rho1
    return out


def theoretical_acf_sq_gjr_garch_1_1(alpha, beta, gamma, nlags):
    """
    Theoretical ACF of eps_t^2 for GJR-GARCH(1,1) with z_t ~ N(0,1):

        h_t = omega + alpha*eps_{t-1}^2 + gamma*eps_{t-1}^2*I_{t-1} + beta*h_{t-1}
        I_{t-1} = 1{eps_{t-1} < 0}

    Write h_t = omega + A_{t-1} h_{t-1} with
        A = beta + (alpha + gamma*I) z^2,  I=1{z<0}.

    Then rho_{eps^2}(k) = p1^(k-1) * rho(1), k>=1, with:
        p1 = E[A]  = beta + alpha + gamma/2
        p2 = E[A^2] = beta^2 + 2*beta*alpha + beta*gamma + 3*alpha^2 + 3*alpha*gamma + (3/2)*gamma^2
        q1 = E[z^2 A] = beta + 3*alpha + (3/2)*gamma

        rho(1) = ((1-p1^2)*q1 - p1*(1-p2)) / (2 + p2 - 3*p1^2)

    Requires p1 < 1 and p2 < 1.
    """
    p1 = beta + alpha + 0.5 * gamma
    p2 = (
        beta * beta
        + 2.0 * beta * alpha
        + beta * gamma
        + 3.0 * alpha * alpha
        + 3.0 * alpha * gamma
        + 1.5 * gamma * gamma
    )
    q1 = beta + 3.0 * alpha + 1.5 * gamma

    if p1 >= 1.0:
        raise ValueError("need beta + alpha + gamma/2 < 1")
    if p2 >= 1.0:
        raise ValueError("need p2 < 1 (finite second moment of h)")

    denom = 2.0 + p2 - 3.0 * p1 * p1
    rho1 = ((1.0 - p1 * p1) * q1 - p1 * (1.0 - p2)) / denom

    out = np.empty(nlags + 1, dtype=float)
    out[0] = 1.0
    for k in range(1, nlags + 1):
        out[k] = (p1 ** (k - 1)) * rho1
    return out


def theoretical_crosscorr_r_r2_nagarch_1_1(omega, alpha, beta, theta, nlags, eh32=None):
    """
    Theoretical cross-correlation out[k] = Corr(eps_t, eps_{t+k}^2) for k=0..nlags
    under NAGARCH(1,1) "2b" with z_t ~ N(0,1).

    Needs E[h^(3/2)] passed as eh32.
    """
    p1 = beta + alpha * (1.0 + theta * theta)
    p2 = (
        beta * beta
        + 2.0 * alpha * beta * (1.0 + theta * theta)
        + alpha * alpha * (3.0 + 6.0 * theta * theta + theta ** 4)
    )

    if p1 >= 1.0:
        raise ValueError("need beta + alpha*(1+theta^2) < 1")
    if p2 >= 1.0:
        raise ValueError("need p2 < 1 (finite second moment of h)")

    if eh32 is None:
        raise ValueError("eh32 is required (estimate E[h^(3/2)] from a long run)")

    mu_h = omega / (1.0 - p1)
    m2 = omega * omega * (1.0 + p1) / ((1.0 - p1) * (1.0 - p2))

    var_r = mu_h
    var_r2 = 3.0 * m2 - mu_h * mu_h

    cov1 = -2.0 * alpha * theta * eh32
    corr1 = cov1 / math.sqrt(var_r * var_r2)

    out = np.empty(nlags + 1, dtype=float)
    out[0] = 0.0
    for k in range(1, nlags + 1):
        out[k] = (p1 ** (k - 1)) * corr1
    return out


def _eh32_approx_from_mu_m2(mu_h, m2, method="taylor2"):
    """
    Approximate E[h^(3/2)] from mu_h=E[h] and m2=E[h^2].

    method:
      "taylor2": second-order Taylor around mu_h using Var(h)=m2-mu_h^2
      "lognormal": match mean and variance with a lognormal approximation
    """
    if mu_h <= 0:
        raise ValueError("mu_h must be > 0")

    var_h = m2 - mu_h * mu_h
    if var_h < 0:
        var_h = 0.0

    if method == "taylor2":
        # E[h^p] approx mu^p + 0.5*p*(p-1)*mu^(p-2)*Var(h), for p=3/2
        return (mu_h ** 1.5) + (3.0 / 8.0) * var_h / math.sqrt(mu_h)

    if method == "lognormal":
        if var_h <= 0:
            return mu_h ** 1.5
        s2 = math.log(1.0 + var_h / (mu_h * mu_h))
        m = math.log(mu_h) - 0.5 * s2
        p = 1.5
        return math.exp(p * m + 0.5 * p * p * s2)

    raise ValueError('unknown method: %s (use "taylor2" or "lognormal")' % method)


def theoretical_crosscorr_r_r2_gjr_garch_1_1(omega, alpha, beta, gamma, nlags, method="taylor2"):
    """
    Theoretical cross-correlation out[k] = Corr(eps_t, eps_{t+k}^2) for k=0..nlags
    under GJR-GARCH(1,1) with z_t ~ N(0,1), using a moment approximation for E[h^(3/2)].

    For k>=1:
        Corr(eps_t, eps_{t+k}^2) = p1^(k-1) * Corr(eps_t, eps_{t+1}^2)

    With normal innovations:
        Cov(eps_t, eps_{t+1}^2) = -gamma * sqrt(2/pi) * E[h^(3/2)].

    E[h^(3/2)] is approximated from mu_h=E[h] and m2=E[h^2] via "method".
    """
    p1 = beta + alpha + 0.5 * gamma
    p2 = (
        beta * beta
        + 2.0 * beta * alpha
        + beta * gamma
        + 3.0 * alpha * alpha
        + 3.0 * alpha * gamma
        + 1.5 * gamma * gamma
    )

    if p1 >= 1.0:
        raise ValueError("need beta + alpha + gamma/2 < 1")
    if p2 >= 1.0:
        raise ValueError("need p2 < 1 (finite second moment of h)")

    mu_h = omega / (1.0 - p1)
    m2 = omega * omega * (1.0 + p1) / ((1.0 - p1) * (1.0 - p2))

    eh32 = _eh32_approx_from_mu_m2(mu_h, m2, method=method)

    var_r = mu_h
    var_r2 = 3.0 * m2 - mu_h * mu_h

    cov1 = -gamma * math.sqrt(2.0 / math.pi) * eh32
    corr1 = cov1 / math.sqrt(var_r * var_r2)

    out = np.empty(nlags + 1, dtype=float)
    out[0] = 0.0
    for k in range(1, nlags + 1):
        out[k] = (p1 ** (k - 1)) * corr1
    return out


def _moments_from_p(omega, p1, p2):
    if omega <= 0:
        raise ValueError("omega must be > 0")
    if p1 >= 1.0:
        raise ValueError("need p1 < 1 for E[h] to exist")
    if p2 >= 1.0:
        raise ValueError("need p2 < 1 for E[h^2] to exist")

    mu_h = omega / (1.0 - p1)
    m2 = omega * omega * (1.0 + p1) / ((1.0 - p1) * (1.0 - p2))
    return mu_h, m2


def _var_r2_kurtosis(mu_h, m2, kappa):
    if kappa <= 0:
        raise ValueError("kappa must be > 0")
    er4 = kappa * m2
    er2 = mu_h
    var_r2 = er4 - er2 * er2
    kurt = er4 / (er2 * er2)
    return var_r2, kurt


def autocov_sq_from_pq(omega, p1, p2, q1, kappa, nlags):
    """
    Return:
      gamma[k] = Cov(r_t^2, r_{t-k}^2), k=0..nlags
      var_r2   = Var(r_t^2)
      kurt     = E[r^4] / E[r^2]^2
      mu_h     = E[h]
      m2       = E[h^2]
    """
    mu_h, m2 = _moments_from_p(omega, p1, p2)
    var_r2, kurt = _var_r2_kurtosis(mu_h, m2, kappa)

    cov1 = omega * mu_h + q1 * m2 - mu_h * mu_h

    gamma = np.empty(nlags + 1, dtype=float)
    gamma[0] = var_r2
    for k in range(1, nlags + 1):
        gamma[k] = (p1 ** (k - 1)) * cov1

    return gamma, var_r2, kurt, mu_h, m2


def pq_garch_1_1(alpha, beta, kappa):
    p1 = alpha + beta
    p2 = beta * beta + 2.0 * alpha * beta + alpha * alpha * kappa
    q1 = beta + alpha * kappa
    return p1, p2, q1


def pq_nagarch_1_1(alpha, beta, theta, kappa):
    p1 = beta + alpha * (1.0 + theta * theta)
    p2 = (
        beta * beta
        + 2.0 * alpha * beta * (1.0 + theta * theta)
        + alpha * alpha * (kappa + 6.0 * theta * theta + theta ** 4)
    )
    q1 = beta + alpha * (kappa + theta * theta)
    return p1, p2, q1


def pq_gjr_garch_1_1(alpha, beta, gamma, kappa):
    p1 = beta + alpha + 0.5 * gamma
    p2 = (
        beta * beta
        + 2.0 * beta * (alpha + 0.5 * gamma)
        + kappa * (alpha * alpha + alpha * gamma + 0.5 * gamma * gamma)
    )
    q1 = beta + kappa * (alpha + 0.5 * gamma)
    return p1, p2, q1


def autocov_sq_garch_1_1(omega, alpha, beta, kappa, nlags):
    p1, p2, q1 = pq_garch_1_1(alpha, beta, kappa)
    return autocov_sq_from_pq(omega, p1, p2, q1, kappa, nlags)


def autocov_sq_nagarch_1_1(omega, alpha, beta, theta, kappa, nlags):
    p1, p2, q1 = pq_nagarch_1_1(alpha, beta, theta, kappa)
    return autocov_sq_from_pq(omega, p1, p2, q1, kappa, nlags)


def autocov_sq_gjr_garch_1_1(omega, alpha, beta, gamma, kappa, nlags):
    p1, p2, q1 = pq_gjr_garch_1_1(alpha, beta, gamma, kappa)
    return autocov_sq_from_pq(omega, p1, p2, q1, kappa, nlags)


def _lognormal_m_s2_from_mu_var(mu, var):
    """
    If h ~ LogNormal(m, s2) then:
        E[h] = exp(m + 0.5*s2) = mu
        Var(h) = mu^2*(exp(s2)-1) = var
    """
    if mu <= 0:
        raise ValueError("mu must be > 0")
    if var <= 0:
        return math.log(mu), 0.0
    s2 = math.log(1.0 + var / (mu * mu))
    m = math.log(mu) - 0.5 * s2
    return m, s2


def _corr_log_from_corr_lognormal(corr_h, s2):
    """
    For lognormal X=exp(U), Y=exp(V) with Var(U)=Var(V)=s2 and Corr(U,V)=rho_u:
        Corr(X,Y) = (exp(rho_u*s2) - 1) / (exp(s2) - 1)

    Solve rho_u from corr_h.
    """
    if s2 <= 0.0:
        return 0.0

    if corr_h > 1.0:
        corr_h = 1.0
    if corr_h < -1.0:
        corr_h = -1.0

    a = math.exp(s2) - 1.0
    x = 1.0 + corr_h * a

    if x <= 0.0:
        x = 1e-300

    return math.log(x) / s2


def _eh_p_lognormal(m, s2, p):
    return math.exp(p * m + 0.5 * p * p * s2)


def _eh_ab_lognormal(m, s2, a, b, rho_u):
    """
    If (U,V) bivariate normal with mean m, var s2, corr rho_u, and
    X=exp(U), Y=exp(V), then:
        E[X^a Y^b] = exp((a+b)m + 0.5*(a^2 + b^2 + 2ab*rho_u)*s2)
    """
    return math.exp((a + b) * m + 0.5 * (a * a + b * b + 2.0 * a * b * rho_u) * s2)


def autocov_abs_from_pq(omega, p1, p2, q1, kappa, nlags, ez_abs=None):
    """
    Tier A approximation for autocovariances of abs returns |r_t|.

    IMPORTANT: for the lognormal mapping we need Corr(h_t, h_{t-k}),
    not Corr(r_t^2, r_{t-k}^2). Using squared-return covariances can
    imply corr(h)>1 and cause clipping/plateaus.
    """
    if ez_abs is None:
        ez_abs = math.sqrt(2.0 / math.pi)

    mu_h, m2 = _moments_from_p(omega, p1, p2)

    var_h = m2 - mu_h * mu_h
    if var_h < 0.0:
        var_h = 0.0

    m, s2 = _lognormal_m_s2_from_mu_var(mu_h, var_h)

    e_sqrt_h = _eh_p_lognormal(m, s2, 0.5)
    mean_abs = ez_abs * e_sqrt_h
    var_abs = mu_h - mean_abs * mean_abs

    gamma_abs = np.empty(nlags + 1, dtype=float)
    gamma_abs[0] = var_abs

    if nlags == 0:
        return gamma_abs, var_abs, mean_abs

    if s2 <= 0.0 or var_h <= 0.0 or var_abs <= 0.0:
        for k in range(1, nlags + 1):
            gamma_abs[k] = 0.0
        return gamma_abs, var_abs, mean_abs

    cov_h1 = omega * mu_h + p1 * m2 - mu_h * mu_h

    for k in range(1, nlags + 1):
        cov_h_k = (p1 ** (k - 1)) * cov_h1
        corr_h_k = cov_h_k / var_h

        rho_u = _corr_log_from_corr_lognormal(corr_h_k, s2)

        e_sqrt_hh = _eh_ab_lognormal(m, s2, 0.5, 0.5, rho_u)
        cov_sqrt_h = e_sqrt_hh - e_sqrt_h * e_sqrt_h

        gamma_abs[k] = (ez_abs * ez_abs) * cov_sqrt_h

    return gamma_abs, var_abs, mean_abs


def acf_abs_from_pq(omega, p1, p2, q1, kappa, nlags, ez_abs=None):
    """
    Tier A approximation for autocorrelations of abs returns |r_t|.

    Returns
    -------
    rho_abs : ndarray, shape (nlags+1,), Corr(|r_t|, |r_{t-k}|)
    var_abs : float, Var(|r_t|)
    mean_abs: float, E|r_t|
    """
    gamma_abs, var_abs, mean_abs = autocov_abs_from_pq(omega, p1, p2, q1, kappa, nlags, ez_abs=ez_abs)

    rho_abs = np.empty(nlags + 1, dtype=float)
    rho_abs[0] = 1.0

    if var_abs <= 0.0:
        for k in range(1, nlags + 1):
            rho_abs[k] = 0.0
        return rho_abs, var_abs, mean_abs

    for k in range(1, nlags + 1):
        rho_abs[k] = gamma_abs[k] / var_abs

    return rho_abs, var_abs, mean_abs


def autocov_abs_garch_1_1(omega, alpha, beta, kappa, nlags, ez_abs=None):
    p1, p2, q1 = pq_garch_1_1(alpha, beta, kappa)
    return autocov_abs_from_pq(omega, p1, p2, q1, kappa, nlags, ez_abs=ez_abs)


def acf_abs_garch_1_1(omega, alpha, beta, kappa, nlags, ez_abs=None):
    p1, p2, q1 = pq_garch_1_1(alpha, beta, kappa)
    return acf_abs_from_pq(omega, p1, p2, q1, kappa, nlags, ez_abs=ez_abs)


def autocov_abs_nagarch_1_1(omega, alpha, beta, theta, kappa, nlags, ez_abs=None):
    p1, p2, q1 = pq_nagarch_1_1(alpha, beta, theta, kappa)
    return autocov_abs_from_pq(omega, p1, p2, q1, kappa, nlags, ez_abs=ez_abs)


def acf_abs_nagarch_1_1(omega, alpha, beta, theta, kappa, nlags, ez_abs=None):
    p1, p2, q1 = pq_nagarch_1_1(alpha, beta, theta, kappa)
    return acf_abs_from_pq(omega, p1, p2, q1, kappa, nlags, ez_abs=ez_abs)


def autocov_abs_gjr_garch_1_1(omega, alpha, beta, gamma, kappa, nlags, ez_abs=None):
    p1, p2, q1 = pq_gjr_garch_1_1(alpha, beta, gamma, kappa)
    return autocov_abs_from_pq(omega, p1, p2, q1, kappa, nlags, ez_abs=ez_abs)


def acf_abs_gjr_garch_1_1(omega, alpha, beta, gamma, kappa, nlags, ez_abs=None):
    p1, p2, q1 = pq_gjr_garch_1_1(alpha, beta, gamma, kappa)
    return acf_abs_from_pq(omega, p1, p2, q1, kappa, nlags, ez_abs=ez_abs)


def figarch_lambda_weights_1_1(alpha, beta, d, n_weights):
    """
    Truncated ARCH(infty) weights lambda_1..lambda_n for FIGARCH(1,d,1).

    Uses the series definition:
        lambda(L) = 1 - (1 - beta L)^(-1) (1 - phi L) (1 - L)^d
        phi = alpha + beta

    We compute coefficients up to n_weights by:
      (1 - L)^d = sum g_k L^k,  g_0=1, g_k = g_{k-1} * ((k-1-d)/k)
      b(L) = (1 - phi L) (1 - L)^d
      c(L) = (1 - beta L)^(-1) b(L),  with recursion c_k = b_k + beta*c_{k-1}
      lambda_k = -c_k for k>=1 (since lambda(L)=1-c(L) and c_0=1)
    """
    if alpha < 0 or beta < 0:
        raise ValueError("alpha and beta must be >= 0")
    if not (0.0 < d < 1.0):
        raise ValueError("need 0 < d < 1 for figarch")
    if n_weights <= 0:
        raise ValueError("n_weights must be >= 1")

    phi = alpha + beta
    if abs(phi) >= 1.0:
        raise ValueError("need |alpha + beta| < 1 for arfima(1,d,1) representation")

    g = np.empty(n_weights + 1, dtype=float)
    g[0] = 1.0
    for k in range(1, n_weights + 1):
        g[k] = g[k - 1] * (((k - 1.0) - d) / k)

    b = np.empty(n_weights + 1, dtype=float)
    b[0] = g[0]
    for k in range(1, n_weights + 1):
        b[k] = g[k] - phi * g[k - 1]

    c = np.empty(n_weights + 1, dtype=float)
    c[0] = b[0]
    for k in range(1, n_weights + 1):
        c[k] = b[k] + beta * c[k - 1]

    lam = np.empty(n_weights, dtype=float)
    for k in range(1, n_weights + 1):
        lam[k - 1] = -c[k]

    return lam


def simulate_figarch_1_1(n, omega, alpha, beta, d, trunc=1000, burn=5000, seed=12345):
    """
    Simulate FIGARCH(1,d,1) using a truncated ARCH(infty) representation:

        eps_t = sqrt(h_t) * z_t,   z_t ~ N(0,1)
        h_t   = omega/(1-beta) + sum_{k=1..trunc} lambda_k * eps_{t-k}^2

    where lambda_k are the FIGARCH(1,d,1) weights returned by
        figarch_lambda_weights_1_1(alpha, beta, d, trunc).

    Notes
    -----
    * This is a truncation-based approximation; long memory implies large trunc may be needed.
    * Requires beta < 1 and sum(lambda_k) < 1 for the truncated unconditional mean to exist.

    Returns
    -------
    eps : ndarray, shape (n,)
    h   : ndarray, shape (n,)
    """
    if omega <= 0:
        raise ValueError("omega must be > 0")
    if alpha < 0 or beta < 0:
        raise ValueError("alpha and beta must be >= 0")
    if beta >= 1.0:
        raise ValueError("need beta < 1")
    if not (0.0 < d < 1.0):
        raise ValueError("need 0 < d < 1 for figarch")
    if trunc <= 0:
        raise ValueError("trunc must be >= 1")
    if burn < 0:
        raise ValueError("burn must be >= 0")
    if n <= 0:
        raise ValueError("n must be >= 1")

    lam = figarch_lambda_weights_1_1(alpha, beta, d, trunc)
    s1 = float(lam.sum())
    if s1 >= 1.0:
        raise ValueError("truncated sum(lambda_k) must be < 1; try smaller trunc or different params")

    rng = np.random.default_rng(seed)
    total = n + burn

    eps = np.empty(total, dtype=float)
    eps2 = np.empty(total, dtype=float)
    h = np.empty(total, dtype=float)

    const = omega / (1.0 - beta)
    mu_h = const / (1.0 - s1)

    h[0] = mu_h
    eps[0] = math.sqrt(h[0]) * rng.standard_normal()
    eps2[0] = eps[0] * eps[0]

    for t in range(1, total):
        k = trunc if t >= trunc else t

        # past = [eps_{t-1}^2, eps_{t-2}^2, ..., eps_{t-k}^2]
        past = eps2[t - k : t][::-1]

        h[t] = const + float(np.dot(lam[:k], past))
        if h[t] <= 0.0:
            raise ValueError("non-positive h encountered at t=%d (h=%.6g)" % (t, h[t]))

        eps[t] = math.sqrt(h[t]) * rng.standard_normal()
        eps2[t] = eps[t] * eps[t]

    return eps[burn:], h[burn:]


def _arfima11_wold(d, ar1, ma1, n_terms):
    """
    Wold coefficients psi[0..n_terms-1] for ARFIMA(1,d,1):

        (1 - ar1 L) (1 - L)^d x_t = (1 + ma1 L) e_t

    so x_t = sum_{j>=0} psi_j e_{t-j}.
    """
    if n_terms <= 0:
        raise ValueError("n_terms must be >= 1")
    if not (0.0 < d < 1.0):
        raise ValueError("need 0 < d < 1")
    if abs(ar1) >= 1.0:
        raise ValueError("need |ar1| < 1 for stability")

    pi = np.empty(n_terms, dtype=float)
    pi[0] = 1.0
    for j in range(1, n_terms):
        pi[j] = pi[j - 1] * ((j - 1.0 + d) / j)

    psi = np.empty(n_terms, dtype=float)
    psi[0] = pi[0]

    for j in range(1, n_terms):
        bj = pi[j] + ma1 * pi[j - 1]
        psi[j] = bj + ar1 * psi[j - 1]

    return psi


def theoretical_acv_sq_figarch_1_1(alpha, beta, d, nlags, n_terms=200_000, sigma2=1.0):
    """
    Approximate autocovariances gamma[k]=Cov(x_t, x_{t+k}) for
    x_t = eps_t^2 - E[eps_t^2] under FIGARCH(1,d,1) via ARFIMA(1,d,1):

        (1 - phi L) (1 - L)^d x_t = (1 - beta L) nu_t,   phi = alpha + beta

    This corresponds to ARFIMA(1,d,1) with:
        ar1 = phi
        ma1 = -beta
    """
    if nlags < 0:
        raise ValueError("nlags must be >= 0")
    if n_terms <= nlags:
        raise ValueError("need n_terms > nlags")
    if sigma2 <= 0.0:
        raise ValueError("sigma2 must be > 0")
    if not (0.0 < d < 0.5):
        raise ValueError("need 0 < d < 0.5 for second-order stationarity")

    phi = alpha + beta
    if abs(phi) >= 1.0:
        raise ValueError("need |alpha + beta| < 1")

    psi = _arfima11_wold(d, phi, -beta, n_terms)

    out = np.empty(nlags + 1, dtype=float)
    for k in range(0, nlags + 1):
        out[k] = sigma2 * float(np.dot(psi[: n_terms - k], psi[k:]))

    return out


def theoretical_acf_sq_figarch_1_1(alpha, beta, d, nlags, n_terms=200_000):
    """
    Approximate theoretical ACF of x_t = eps_t^2 - E[eps_t^2] for FIGARCH(1,d,1),
    using the ARFIMA(1,d,1) representation:

        (1 - phi L) (1 - L)^d x_t = (1 - beta L) nu_t,
        phi = alpha + beta.

    This is ARFIMA(1,d,1) with:
        ar1 = phi
        ma1 = -beta

    The ACF is computed from the Wold coefficients psi_j:
        x_t = sum_{j>=0} psi_j nu_{t-j},  Var(nu)=1

        gamma(k) = Cov(x_t, x_{t+k}) = sum_{j>=0} psi_j psi_{j+k}
        rho(k) = gamma(k)/gamma(0)
    """
    if nlags < 0:
        raise ValueError("nlags must be >= 0")
    if n_terms <= nlags:
        raise ValueError("need n_terms > nlags")
    if not (0.0 < d < 0.5):
        raise ValueError("need 0 < d < 0.5 for second-order stationarity")

    phi = alpha + beta
    if abs(phi) >= 1.0:
        raise ValueError("need |alpha + beta| < 1")

    psi = _arfima11_wold(d, phi, -beta, n_terms)

    gamma0 = float(np.dot(psi, psi))
    if gamma0 <= 0.0:
        raise ValueError("non-positive gamma0 (increase n_terms or check parameters)")

    out = np.empty(nlags + 1, dtype=float)
    out[0] = 1.0
    for k in range(1, nlags + 1):
        out[k] = float(np.dot(psi[: n_terms - k], psi[k:])) / gamma0

    return out


def autocov_abs_figarch_1_1(
    omega,
    alpha,
    beta,
    d,
    kappa,
    nlags,
    trunc=1000,
    n_terms=200_000,
    ez_abs=None,
):
    """
    Tier A (delta/lognormal) approximation for autocovariances of |eps_t| under FIGARCH(1,d,1).

    Model (simulation uses the truncated arch(infty) representation):
        eps_t = sqrt(h_t) z_t,  E[z^2]=1, E[z^4]=kappa, symmetric
        h_t   = omega/(1-beta) + sum_{j>=1} lambda_j eps_{t-j}^2

    For the dependence structure of eps^2 we use the arfima(1,d,1) representation:
        (1 - phi L) (1 - L)^d x_t = (1 - beta L) nu_t
        x_t = eps_t^2 - E[eps_t^2],  phi = alpha + beta
    """
    if ez_abs is None:
        ez_abs = math.sqrt(2.0 / math.pi)

    if kappa <= 1.0:
        raise ValueError("need kappa > 1")

    lam = figarch_lambda_weights_1_1(alpha, beta, d, trunc)
    s1 = float(lam.sum())
    if s1 >= 1.0:
        raise ValueError("truncated sum(lambda_k) must be < 1; try smaller trunc or different params")

    if beta >= 1.0:
        raise ValueError("need beta < 1 for omega/(1-beta)")

    const = omega / (1.0 - beta)
    mu_h = const / (1.0 - s1)

    base_gamma = theoretical_acv_sq_figarch_1_1(alpha, beta, d, nlags, n_terms=n_terms, sigma2=1.0)
    base0 = float(base_gamma[0])

    denom = kappa - (kappa - 1.0) * base0
    if denom <= 0.0:
        raise ValueError("non-positive denom in m2 identification; adjust params/trunc/n_terms")

    m2 = (mu_h * mu_h) / denom
    var_h = m2 - mu_h * mu_h
    if var_h < 0.0:
        var_h = 0.0

    sigma2 = (kappa - 1.0) * m2  # Var(nu_t)

    m, s2 = _lognormal_m_s2_from_mu_var(mu_h, var_h)

    e_sqrt_h = _eh_p_lognormal(m, s2, 0.5)
    mean_abs = ez_abs * e_sqrt_h
    var_abs = mu_h - mean_abs * mean_abs

    gamma_abs = np.empty(nlags + 1, dtype=float)
    gamma_abs[0] = var_abs

    if nlags == 0:
        return gamma_abs, var_abs, mean_abs

    if s2 <= 0.0 or var_h <= 0.0 or var_abs <= 0.0:
        for k in range(1, nlags + 1):
            gamma_abs[k] = 0.0
        return gamma_abs, var_abs, mean_abs

    for k in range(1, nlags + 1):
        cov_h_k = sigma2 * float(base_gamma[k])
        corr_h_k = cov_h_k / var_h

        rho_u = _corr_log_from_corr_lognormal(corr_h_k, s2)
        e_sqrt_hh = _eh_ab_lognormal(m, s2, 0.5, 0.5, rho_u)

        cov_sqrt_h = e_sqrt_hh - e_sqrt_h * e_sqrt_h
        gamma_abs[k] = (ez_abs * ez_abs) * cov_sqrt_h

    return gamma_abs, var_abs, mean_abs


def acf_abs_figarch_1_1(
    omega,
    alpha,
    beta,
    d,
    kappa,
    nlags,
    trunc=1000,
    n_terms=200_000,
    ez_abs=None,
):
    """
    Tier A (delta/lognormal) approximation for acf of |eps_t| under FIGARCH(1,d,1).

    Returns
    -------
    rho_abs : ndarray, shape (nlags+1,), Corr(|eps_t|, |eps_{t-k}|)
    var_abs : float, Var(|eps_t|)
    mean_abs: float, E|eps_t|
    """
    gamma_abs, var_abs, mean_abs = autocov_abs_figarch_1_1(
        omega,
        alpha,
        beta,
        d,
        kappa,
        nlags,
        trunc=trunc,
        n_terms=n_terms,
        ez_abs=ez_abs,
    )

    rho_abs = np.empty(nlags + 1, dtype=float)
    rho_abs[0] = 1.0

    if var_abs <= 0.0:
        for k in range(1, nlags + 1):
            rho_abs[k] = 0.0
        return rho_abs, var_abs, mean_abs

    for k in range(1, nlags + 1):
        rho_abs[k] = gamma_abs[k] / var_abs

    return rho_abs, var_abs, mean_abs

def _std_norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _egarch_normal_constants():
    ez_abs = math.sqrt(2.0 / math.pi)
    var_abs = 1.0 - 2.0 / math.pi
    phi0 = 1.0 / math.sqrt(2.0 * math.pi)
    return ez_abs, var_abs, phi0

def _egarch_expx_moment_normal(omega, alpha, beta, gamma, s, n_terms=2000, tol=1e-12):
    """
    Exact (up to truncation) moment for EGARCH log-variance:

        x = omega/(1-beta) + sum_{j>=0} beta^j u_{-1-j}
        u = alpha(|z|-E|z|) + gamma z

    E[exp(s x)] = exp(s*omega/(1-beta)) * prod_{j>=0} M_u(s beta^j)

    Uses truncation when |s beta^j| becomes small.
    """
    if abs(beta) >= 1.0:
        raise ValueError("need |beta| < 1")
    if n_terms <= 0:
        raise ValueError("n_terms must be >= 1")

    log_out = s * omega / (1.0 - beta)

    t = s
    for _j in range(n_terms):
        m = _egarch_m_u_normal(t, alpha, gamma)
        if m <= 0.0:
            raise ValueError("non-positive M_u encountered (check params)")
        log_out += math.log(m)

        t *= beta
        if abs(t) < tol:
            break
    return math.exp(log_out)

def _egarch_j_pos_normal(t):
    """
    J(t) = int_0^inf u^2 exp(t u) phi(u) du, phi standard normal pdf.

    Closed form (differentiate I(t)):
      J(t) = t*phi(0) + (1 + t^2) * exp(t^2/2) * Phi(t)
    """
    _ez_abs, _var_abs, phi0 = _egarch_normal_constants()
    return t * phi0 + (1.0 + t * t) * math.exp(0.5 * t * t) * _std_norm_cdf(t)


def _egarch_a_u_normal(t, alpha, gamma):
    """
    A_u(t) = E[ |z| * exp( t * ( alpha(|z|-E|z|) + gamma z ) ) ] for z~N(0,1).
    """
    ez_abs, _var_abs, _phi0 = _egarch_normal_constants()
    a = t * alpha
    g = t * gamma
    # E[ |z| exp(a|z| + g z) ] = I(a+g) + I(a-g)
    return math.exp(-a * ez_abs) * (_egarch_i_pos_normal(a + g) + _egarch_i_pos_normal(a - g))

def _egarch_i_pos_normal(t):
    """
    I(t) = int_0^inf u exp(t u) phi(u) du, for phi standard normal pdf.
    Closed form: I(t) = phi(0) + t exp(t^2/2) Phi(t).
    """
    _ez_abs, _var_abs, phi0 = _egarch_normal_constants()
    return phi0 + t * math.exp(0.5 * t * t) * _std_norm_cdf(t)
    
def _egarch_s_u_normal(t, alpha, gamma):
    """
    S_u(t) = E[ z * exp( t * ( alpha(|z|-E|z|) + gamma z ) ) ] for z~N(0,1).
    """
    ez_abs, _var_abs, _phi0 = _egarch_normal_constants()
    a = t * alpha
    g = t * gamma

    return math.exp(-a * ez_abs) * (_egarch_i_pos_normal(a + g) - _egarch_i_pos_normal(a - g))
    
def _egarch_q_u_normal(t, alpha, gamma):
    """
    Q_u(t) = E[ z^2 * exp( t * ( alpha(|z|-E|z|) + gamma z ) ) ] for z~N(0,1).
    """
    ez_abs, _var_abs, _phi0 = _egarch_normal_constants()
    a = t * alpha
    g = t * gamma
    # E[ z^2 exp(a|z| + g z) ] = J(a+g) + J(a-g)
    return math.exp(-a * ez_abs) * (_egarch_j_pos_normal(a + g) + _egarch_j_pos_normal(a - g))


def theoretical_acf_sq_egarch_1_1(alpha, beta, gamma, kappa, nlags, n_terms=2000, tol=1e-12):
    """
    Exact (up to truncation) ACF of eps_t^2 for EGARCH(1,1) with NORMAL z_t.

    Note: because M_u / Q_u are normal-specific, mixing kappa != 3 with these
    formulas is inconsistent. If you want non-normal, you need the matching
    M_u and Q_u for that distribution.
    """
    if nlags < 0:
        raise ValueError("nlags must be >= 0")
    if abs(beta) >= 1.0:
        raise ValueError("need |beta| < 1")
    if kappa <= 1.0:
        raise ValueError("need kappa > 1")
    if abs(kappa - 3.0) > 1e-12:
        raise ValueError("for these normal-based formulas, use kappa=3")

    # ACF does not depend on omega (scale cancels), so omega=0 is fine.
    acov, var_r2, _kurt, _mu_h, _eh2 = autocov_sq_egarch_1_1(
        0.0, alpha, beta, gamma, kappa, nlags, n_terms=n_terms, tol=tol
    )

    out = np.empty(nlags + 1, dtype=float)
    out[0] = 1.0
    if var_r2 <= 0.0:
        for k in range(1, nlags + 1):
            out[k] = 0.0
        return out

    for k in range(1, nlags + 1):
        out[k] = acov[k] / var_r2
    return out


def autocov_sq_egarch_1_1(omega, alpha, beta, gamma, kappa, nlags, n_terms=2000, tol=1e-12):
    """
    Exact (up to truncation) autocovariances of eps_t^2 for EGARCH(1,1) with NORMAL z_t.

    Returns
    -------
    gamma : ndarray, shape (nlags+1,), Cov(eps_t^2, eps_{t-k}^2)
    var_r2: float, Var(eps_t^2)
    kurt : float, E[eps^4] / E[eps^2]^2
    mu_h : float, E[h] where h=variance
    m2   : float, E[h^2]
    """
    if nlags < 0:
        raise ValueError("nlags must be >= 0")
    if abs(beta) >= 1.0:
        raise ValueError("need |beta| < 1")
    if kappa <= 1.0:
        raise ValueError("need kappa > 1")
    if abs(kappa - 3.0) > 1e-12:
        raise ValueError("for these normal-based formulas, use kappa=3")

    eh = _egarch_expx_moment_normal(omega, alpha, beta, gamma, 1.0, n_terms=n_terms, tol=tol)
    eh2 = _egarch_expx_moment_normal(omega, alpha, beta, gamma, 2.0, n_terms=n_terms, tol=tol)

    var_r2 = kappa * eh2 - eh * eh
    if var_r2 <= 0.0:
        raise ValueError("non-positive Var(eps^2) (check params)")

    kurt = (kappa * eh2) / (eh * eh)
    mu_h = eh

    out = np.empty(nlags + 1, dtype=float)
    out[0] = var_r2

    inv_1mb = 1.0 / (1.0 - beta)

    # log(prod_{j=0..k-2} M_u(beta^j)) updated incrementally in k
    log_future = 0.0

    for k in range(1, nlags + 1):
        bk = beta ** k

        # update future product for this k (adds j=k-2 term)
        if k >= 2:
            t = beta ** (k - 2)
            m = _egarch_m_u_normal(t, alpha, gamma)
            if m <= 0.0:
                raise ValueError("non-positive M_u encountered (check params)")
            log_future += math.log(m)

        future = math.exp(log_future)

        # constant from the recursion expansion of x_{t+k}
        const = math.exp(omega * (1.0 - bk) * inv_1mb)

        # common-past moment: E[ exp( (1+beta^k) x_t ) ]
        ex = _egarch_expx_moment_normal(omega, alpha, beta, gamma, 1.0 + bk, n_terms=n_terms, tol=tol)

        # correlated z_t term: E[ z^2 * exp(beta^{k-1} u) ]
        q = _egarch_q_u_normal(beta ** (k - 1), alpha, gamma)

        cross = const * ex * future * q  # E[ eps_t^2 eps_{t+k}^2 ]
        out[k] = cross - eh * eh

    return out, var_r2, kurt, mu_h, eh2


def autocov_abs_egarch_1_1(omega, alpha, beta, gamma, nlags, ez_abs=None, n_terms=2000, tol=1e-12):
    """
    autocovariance of |eps_t| for egarch(1,1) with normal innovations.

    this version is exact (up to truncation): it accounts for the dependence of h_t on past z's,
    so it does NOT factor into ez_abs^2 * cov(sqrt(h)).
    """
    if ez_abs is None:
        ez_abs = math.sqrt(2.0 / math.pi)

    # unconditional moments
    eh12 = _egarch_expx_moment_normal(omega, alpha, beta, gamma, 0.5, n_terms=n_terms, tol=tol)
    eh1 = _egarch_expx_moment_normal(omega, alpha, beta, gamma, 1.0, n_terms=n_terms, tol=tol)

    mean_abs = ez_abs * eh12
    var_abs = eh1 - mean_abs * mean_abs

    out = np.zeros(nlags + 1)
    out[0] = var_abs

    for k in range(1, nlags + 1):
        joint = ez_abs * _egarch_expx_joint_moment_absz_lag_normal(
            omega, alpha, beta, gamma, 0.5, 0.5, k, n_terms=n_terms, tol=tol
        )
        out[k] = joint - mean_abs * mean_abs

    return out, var_abs, mean_abs

def acf_abs_egarch_1_1(omega, alpha, beta, gamma, nlags, ez_abs=None, n_terms=2000, tol=1e-12):
    """acf(|eps_t|) for egarch(1,1) with normal innovations."""
    acov, var_abs, mean_abs = autocov_abs_egarch_1_1(
        omega, alpha, beta, gamma, nlags, ez_abs=ez_abs, n_terms=n_terms, tol=tol
    )
    out = acov / var_abs
    out[0] = 1.0
    return out, var_abs, mean_abs

def _egarch_m_u_normal(t, alpha, gamma):
    """
    M_u(t) = E[ exp( t * ( alpha(|z|-E|z|) + gamma z ) ) ] for z~N(0,1).
    """
    ez_abs, _var_abs, _phi0 = _egarch_normal_constants()
    a = t * alpha
    g = t * gamma

    # E[exp(a|z| + g z)] = exp((a+g)^2/2) Phi(a+g) + exp((g-a)^2/2) Phi(a-g)
    term1 = math.exp(0.5 * (a + g) * (a + g)) * _std_norm_cdf(a + g)
    term2 = math.exp(0.5 * (g - a) * (g - a)) * _std_norm_cdf(a - g)

    return math.exp(-a * ez_abs) * (term1 + term2)

def simulate_egarch_1_1(n, omega, alpha, beta, gamma, burn=5000, seed=12345, ez_abs=None):
    """
    Simulate EGARCH(1,1) (Nelson-style) with normal innovations:

        eps_t = sqrt(h_t) * z_t,  z_t ~ N(0,1)
        x_t   = log(h_t)
        x_t   = omega + beta * x_{t-1} + alpha * (|z_{t-1}| - E|z|) + gamma * z_{t-1}

    Parameters
    ----------
    omega, alpha, beta, gamma : float
    ez_abs : float or None
        If None, uses E|Z| for Z~N(0,1) = sqrt(2/pi).

    Returns
    -------
    eps : ndarray, shape (n,)
    h   : ndarray, shape (n,)
    """
    if n <= 0:
        raise ValueError("n must be >= 1")
    if burn < 0:
        raise ValueError("burn must be >= 0")
    if abs(beta) >= 1.0:
        raise ValueError("need |beta| < 1 for covariance stationarity of log(h)")

    if ez_abs is None:
        ez_abs, _var_abs, _phi0 = _egarch_normal_constants()

    rng = np.random.default_rng(seed)
    total = n + burn

    eps = np.empty(total, dtype=float)
    h = np.empty(total, dtype=float)
    x = np.empty(total, dtype=float)

    # start at unconditional mean of x_t = log(h_t)
    x[0] = omega / (1.0 - beta)
    h[0] = math.exp(x[0])

    z_prev = rng.standard_normal()
    eps[0] = math.sqrt(h[0]) * z_prev

    for t in range(1, total):
        x[t] = omega + beta * x[t - 1] + alpha * (abs(z_prev) - ez_abs) + gamma * z_prev
        h[t] = math.exp(x[t])

        z_prev = rng.standard_normal()
        eps[t] = math.sqrt(h[t]) * z_prev

    return eps[burn:], h[burn:]

def theoretical_crosscorr_r_r2_egarch_1_1(omega, alpha, beta, gamma, nlags, kappa=3.0, n_terms=2000):
    """
    Theoretical cross-correlation out[k] = Corr(eps_t, eps_{t+k}^2) for k=0..nlags
    under EGARCH(1,1) with normal innovations, using truncated-product formulas.

    For k>=1:
      Cov(eps_t, eps_{t+k}^2) = E[ eps_t * h_{t+k} ]  (since E[z_{t+k}^2]=1)

    With the EGARCH recursion for x=log(h), one can write:
      x_{t+k} = omega*(1-beta^k)/(1-beta) + beta^k x_t + sum_{j=0..k-1} beta^j u_{t+k-1-j},
      u_t = alpha(|z_t|-E|z|) + gamma z_t.

    This yields (k>=1):
      Cov(eps_t, eps_{t+k}^2)
        = exp(omega*(1-beta^k)/(1-beta))
          * E[ exp((beta^k + 1/2) x_t) ]
          * prod_{j=0..k-2} M_u(beta^j)
          * E[ z * exp(beta^{k-1} u) ].

    E[exp(s x)] is computed by truncated infinite product.

    Note: because M_u / S_u are normal-specific, using kappa != 3 with these
    formulas is inconsistent. For these normal-based formulas, use kappa=3.
    """
    if nlags < 0:
        raise ValueError("nlags must be >= 0")
    if abs(beta) >= 1.0:
        raise ValueError("need |beta| < 1")
    if kappa <= 1.0:
        raise ValueError("need kappa > 1")
    if abs(kappa - 3.0) > 1e-12:
        raise ValueError("for these normal-based formulas, use kappa=3")

    eh = _egarch_expx_moment_normal(omega, alpha, beta, gamma, 1.0, n_terms=n_terms)
    eh2 = _egarch_expx_moment_normal(omega, alpha, beta, gamma, 2.0, n_terms=n_terms)

    var_r = eh
    var_r2 = kappa * eh2 - eh * eh
    if var_r2 <= 0.0:
        raise ValueError("non-positive Var(eps^2) (check params)")

    denom = math.sqrt(var_r * var_r2)

    out = np.empty(nlags + 1, dtype=float)
    out[0] = 0.0

    if nlags == 0:
        return out

    inv_1mb = 1.0 / (1.0 - beta)

    # log(prod_{j=0..k-2} M_u(beta^j)) updated incrementally in k
    log_future = 0.0

    for k in range(1, nlags + 1):
        bk = beta ** k

        # update future product for this k (adds j=k-2 term)
        if k >= 2:
            t = beta ** (k - 2)
            m = _egarch_m_u_normal(t, alpha, gamma)
            if m <= 0.0:
                raise ValueError("non-positive M_u encountered (check params)")
            log_future += math.log(m)

        future = math.exp(log_future)

        # intercept contribution
        scale = math.exp(omega * (1.0 - bk) * inv_1mb)

        # moment in x_t with exponent (beta^k + 1/2)
        s_x = bk + 0.5
        ex = _egarch_expx_moment_normal(omega, alpha, beta, gamma, s_x, n_terms=n_terms)

        # z-linked term at lag k: E[ z * exp(beta^{k-1} u) ]
        su = _egarch_s_u_normal(beta ** (k - 1), alpha, gamma)

        cov = scale * ex * future * su
        out[k] = cov / denom

    return out

def _egarch_expx_joint_moment_normal(omega, alpha, beta, gamma, p, q, k, n_terms=5000, tol=1e-12):
    """
    Compute E[ exp( p*x_t + q*x_{t-k} ) ] for EGARCH(1,1) with normal z,
    where x_t = log(h_t) satisfies:
        x_t = omega + beta*x_{t-1} + u_{t-1},
        u_t = alpha(|z_t|-E|z|) + gamma z_t.

    Using the stationary expansion:
        x_t = omega/(1-beta) + sum_{j>=0} beta^j u_{t-1-j}

    For k>=1:
        p*x_t + q*x_{t-k}
        = (p+q)*omega/(1-beta)
          + sum_{j=0..k-1} (p*beta^j) u_{t-1-j}
          + sum_{m>=0} ( (p*beta^k + q) * beta^m ) u_{t-1-k-m}

    So:
        E[exp(p*x_t + q*x_{t-k})]
          = exp((p+q)*omega/(1-beta))
            * prod_{j=0..k-1} M_u(p*beta^j)
            * prod_{m>=0} M_u((p*beta^k + q)*beta^m)

    For k=0, reduces to E[exp((p+q)*x)].
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    if abs(beta) >= 1.0:
        raise ValueError("need |beta| < 1")
    if n_terms <= 0:
        raise ValueError("n_terms must be >= 1")

    if k == 0:
        return _egarch_expx_moment_normal(omega, alpha, beta, gamma, p + q, n_terms=n_terms, tol=tol)

    log_out = (p + q) * omega / (1.0 - beta)

    # finite product: j=0..k-1 of M_u(p*beta^j)
    t = p
    for _j in range(k):
        m = _egarch_m_u_normal(t, alpha, gamma)
        if m <= 0.0:
            raise ValueError("non-positive M_u encountered (check params)")
        log_out += math.log(m)
        t *= beta

    # infinite product: m>=0 of M_u((p*beta^k + q)*beta^m)
    t = (p * (beta ** k) + q)
    for _m in range(n_terms):
        mval = _egarch_m_u_normal(t, alpha, gamma)
        if mval <= 0.0:
            raise ValueError("non-positive M_u encountered (check params)")
        log_out += math.log(mval)

        t *= beta
        if abs(t) < tol:
            break

    return math.exp(log_out)



def _egarch_expx_joint_moment_absz_lag_normal(omega, alpha, beta, gamma, p, q, k, n_terms=2000, tol=1e-12):
    """
    return E[ |z_{t-k}| * exp(p*x_t + q*x_{t-k}) ] for egarch(1,1) with normal innovations.

    this is the same infinite-product construction as _egarch_expx_joint_moment_normal(), but replaces
    the single factor for u_{t-k} with E[ |z| * exp(t*u) ].
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    log_out = (p + q) * omega / (1 - beta)

    # finite part from u_{t-1}, ..., u_{t-k}; last term corresponds to u_{t-k}
    t = p
    for j in range(k):
        if j == k - 1:
            log_out += math.log(_egarch_a_u_normal(t, alpha, gamma))
        else:
            log_out += math.log(_egarch_m_u_normal(t, alpha, gamma))
        t *= beta

    # shared infinite tail
    t = p * (beta**k) + q
    for _ in range(n_terms):
        log_out += math.log(_egarch_m_u_normal(t, alpha, gamma))
        t *= beta
        if abs(t) < tol:
            break

    return math.exp(log_out)


def theoretical_acf_sq_egarch_1_1(alpha, beta, gamma, kappa, nlags, n_terms=5000, tol=1e-12):
    """
    Theoretical ACF of eps_t^2 for EGARCH(1,1) with normal innovations.

    Uses truncated exact product moments (not the lognormal tier-a approx).
    Note: ACF does not depend on omega (scale cancels), so omega is set to 0.
    """
    if nlags < 0:
        raise ValueError("nlags must be >= 0")
    if abs(beta) >= 1.0:
        raise ValueError("need |beta| < 1")
    if kappa <= 1.0:
        raise ValueError("need kappa > 1")

    omega = 0.0

    eh = _egarch_expx_moment_normal(omega, alpha, beta, gamma, 1.0, n_terms=n_terms, tol=tol)
    eh2 = _egarch_expx_moment_normal(omega, alpha, beta, gamma, 2.0, n_terms=n_terms, tol=tol)

    var_r2 = kappa * eh2 - eh * eh
    if var_r2 <= 0.0:
        raise ValueError("non-positive Var(eps^2) (check params)")

    out = np.empty(nlags + 1, dtype=float)
    out[0] = 1.0

    for k in range(1, nlags + 1):
        ehh = _egarch_expx_joint_moment_normal(omega, alpha, beta, gamma, 1.0, 1.0, k, n_terms=n_terms, tol=tol)
        cov = ehh - eh * eh
        out[k] = cov / var_r2

    return out


def autocov_sq_egarch_1_1(omega, alpha, beta, gamma, kappa, nlags, n_terms=5000, tol=1e-12):
    """
    Theoretical autocovariances of eps_t^2 for EGARCH(1,1) with normal innovations,
    using truncated exact product moments.

    Returns
    -------
    gamma : ndarray, shape (nlags+1,), Cov(eps_t^2, eps_{t-k}^2)
    var_r2: float, Var(eps_t^2)
    kurt : float, E[eps^4] / E[eps^2]^2
    mu_h : float, E[h]
    m2   : float, E[h^2]
    """
    if nlags < 0:
        raise ValueError("nlags must be >= 0")
    if abs(beta) >= 1.0:
        raise ValueError("need |beta| < 1")
    if kappa <= 1.0:
        raise ValueError("need kappa > 1")

    eh = _egarch_expx_moment_normal(omega, alpha, beta, gamma, 1.0, n_terms=n_terms, tol=tol)
    eh2 = _egarch_expx_moment_normal(omega, alpha, beta, gamma, 2.0, n_terms=n_terms, tol=tol)

    mu_h = eh
    m2 = eh2

    var_r2 = kappa * m2 - mu_h * mu_h
    if var_r2 <= 0.0:
        raise ValueError("non-positive Var(eps^2) (check params)")

    kurt = (kappa * m2) / (mu_h * mu_h)

    gamma_out = np.empty(nlags + 1, dtype=float)
    gamma_out[0] = var_r2

    for k in range(1, nlags + 1):
        ehh = _egarch_expx_joint_moment_normal(omega, alpha, beta, gamma, 1.0, 1.0, k, n_terms=n_terms, tol=tol)
        gamma_out[k] = ehh - mu_h * mu_h

    return gamma_out, var_r2, kurt, mu_h, m2


def autocov_abs_egarch_1_1(omega, alpha, beta, gamma, nlags, ez_abs=None, n_terms=2000, tol=1e-12):
    """
    autocovariance of |eps_t| for egarch(1,1) with normal innovations.

    this version is exact (up to truncation): it accounts for the dependence of h_t on past z's,
    so it does NOT factor into ez_abs^2 * cov(sqrt(h)).
    """
    if ez_abs is None:
        ez_abs = math.sqrt(2.0 / math.pi)

    # unconditional moments
    eh12 = _egarch_expx_moment_normal(omega, alpha, beta, gamma, 0.5, n_terms=n_terms, tol=tol)
    eh1 = _egarch_expx_moment_normal(omega, alpha, beta, gamma, 1.0, n_terms=n_terms, tol=tol)

    mean_abs = ez_abs * eh12
    var_abs = eh1 - mean_abs * mean_abs

    out = np.zeros(nlags + 1)
    out[0] = var_abs

    for k in range(1, nlags + 1):
        joint = ez_abs * _egarch_expx_joint_moment_absz_lag_normal(
            omega, alpha, beta, gamma, 0.5, 0.5, k, n_terms=n_terms, tol=tol
        )
        out[k] = joint - mean_abs * mean_abs

    return out, var_abs, mean_abs

def acf_abs_egarch_1_1(omega, alpha, beta, gamma, nlags, ez_abs=None, n_terms=2000, tol=1e-12):
    """acf(|eps_t|) for egarch(1,1) with normal innovations."""
    acov, var_abs, mean_abs = autocov_abs_egarch_1_1(
        omega, alpha, beta, gamma, nlags, ez_abs=ez_abs, n_terms=n_terms, tol=tol
    )
    out = acov / var_abs
    out[0] = 1.0
    return out, var_abs, mean_abs




# ---------------------------------------------------------------------
# log-ar(1) stochastic volatility model (sv) - case (a): no leverage
#   x_t = omega + phi * x_{t-1} + sigma * eta_t
#   h_t = exp(x_t)
#   eps_t = sqrt(h_t) * z_t
# where eta_t and z_t are iid with mean 0, var 1 and independent.
# ---------------------------------------------------------------------

def simulate_sv_ar1_case_a(n, omega, phi, sigma, burn=0, seed=None):
    """
    simulate log-ar(1) stochastic volatility (case a, no leverage)

    returns
    -------
    eps : ndarray, shape (n,)
        returns
    h : ndarray, shape (n,)
        conditional variance h_t = exp(x_t)
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if burn < 0:
        raise ValueError("burn must be nonnegative")
    if abs(phi) >= 1:
        raise ValueError("require |phi| < 1 for stationarity")
    if sigma < 0:
        raise ValueError("sigma must be nonnegative")

    rng = np.random.default_rng(seed)

    n_tot = n + burn
    x = np.empty(n_tot, dtype=float)

    mean_x = omega / (1.0 - phi)
    var_x = (sigma * sigma) / (1.0 - phi * phi) if sigma != 0 else 0.0

    # start at stationary mean (burn handles transients)
    x[0] = mean_x
    eta = rng.standard_normal(n_tot)
    for t in range(1, n_tot):
        x[t] = omega + phi * x[t - 1] + sigma * eta[t]

    h = np.exp(x)
    z = rng.standard_normal(n_tot)
    eps = np.sqrt(h) * z

    if burn:
        return eps[burn:], h[burn:]
    return eps, h


def moments_logh_sv_ar1_case_a(omega, phi, sigma):
    """
    moments of x_t (log-variance) in sv ar(1) case a (no leverage)

    returns
    -------
    mean_x : float
    var_x : float
    """
    if abs(phi) >= 1:
        raise ValueError("require |phi| < 1 for stationarity")
    if sigma < 0:
        raise ValueError("sigma must be nonnegative")

    mean_x = omega / (1.0 - phi)
    var_x = (sigma * sigma) / (1.0 - phi * phi) if sigma != 0 else 0.0
    return mean_x, var_x


def moments_h_sv_ar1_case_a(omega, phi, sigma):
    """
    moments of h_t = exp(x_t) for sv ar(1) case a (no leverage)

    returns
    -------
    eh : float
        E[h_t]
    eh2 : float
        E[h_t^2]
    """
    mean_x, var_x = moments_logh_sv_ar1_case_a(omega, phi, sigma)
    eh = math.exp(mean_x + 0.5 * var_x)
    eh2 = math.exp(2.0 * mean_x + 2.0 * var_x)
    return eh, eh2


def var_r2_kurtosis_sv_ar1_case_a(omega, phi, sigma, kappa=3.0):
    """
    variance of eps^2 and kurtosis of eps for sv ar(1) case a (no leverage)

    parameters
    ----------
    kappa : float
        E[z^4] for standardized return innovation z_t. for normal, kappa=3.

    returns
    -------
    var_r2 : float
        Var(eps^2)
    kurt : float
        kurtosis of eps (E[eps^4]/E[eps^2]^2)
    """
    eh, eh2 = moments_h_sv_ar1_case_a(omega, phi, sigma)
    return _var_r2_kurtosis(eh, eh2, kappa)


def theoretical_acf_sq_sv_ar1_case_a(nlags, omega, phi, sigma, kappa=3.0):
    """
    theoretical acf of eps^2 for sv ar(1) case a (no leverage)

    returns array of length nlags with entries for lags 1..nlags.
    """
    if nlags <= 0:
        raise ValueError("nlags must be positive")
    if abs(phi) >= 1:
        raise ValueError("require |phi| < 1 for stationarity")
    if sigma < 0:
        raise ValueError("sigma must be nonnegative")
    if kappa <= 1:
        raise ValueError("kappa must exceed 1 (E[z^4])")

    mean_x, var_x = moments_logh_sv_ar1_case_a(omega, phi, sigma)

    # denom = Var(eps^2) / E[h]^2
    # Var(eps^2) = kappa * E[h^2] - E[h]^2 = E[h]^2 * (kappa * exp(var_x) - 1)
    denom = kappa * math.exp(var_x) - 1.0
    if denom <= 0:
        # should not happen for valid kappa and var_x, but guard anyway
        raise ValueError("invalid parameters: denom <= 0")

    acf = np.empty(nlags, dtype=float)
    for k in range(1, nlags + 1):
        acf[k - 1] = (math.exp(var_x * (phi ** k)) - 1.0) / denom
    return acf


def autocov_sq_sv_ar1_case_a(nlags, omega, phi, sigma, kappa=3.0):
    """
    theoretical autocovariance of eps^2 for sv ar(1) case a (no leverage)

    returns array of length nlags+1 with entries for lags 0..nlags.
    """
    if nlags < 0:
        raise ValueError("nlags must be nonnegative")
    if abs(phi) >= 1:
        raise ValueError("require |phi| < 1 for stationarity")
    if sigma < 0:
        raise ValueError("sigma must be nonnegative")
    if kappa <= 1:
        raise ValueError("kappa must exceed 1 (E[z^4])")

    mean_x, var_x = moments_logh_sv_ar1_case_a(omega, phi, sigma)
    eh = math.exp(mean_x + 0.5 * var_x)
    eh2 = math.exp(2.0 * mean_x + 2.0 * var_x)

    var_r2, _kurt = _var_r2_kurtosis(eh, eh2, kappa)

    acv = np.empty(nlags + 1, dtype=float)
    acv[0] = var_r2

    # for k >= 1:
    # Cov(eps_t^2, eps_{t+k}^2) = Cov(h_t, h_{t+k})
    # = E[h_t h_{t+k}] - E[h]^2
    # with joint normal x, E[exp(x_t + x_{t+k})] = exp(2 mean_x + var_x (1 + phi^k))
    eh_sq = eh * eh
    for k in range(1, nlags + 1):
        e_hh = math.exp(2.0 * mean_x + var_x * (1.0 + (phi ** k)))
        acv[k] = e_hh - eh_sq
    return acv


# aliases (case a)
def simulate_sv_ar1(n, omega, phi, sigma, burn=0, seed=None):
    return simulate_sv_ar1_case_a(n=n, omega=omega, phi=phi, sigma=sigma, burn=burn, seed=seed)

def moments_logh_sv_ar1(omega, phi, sigma):
    return moments_logh_sv_ar1_case_a(omega=omega, phi=phi, sigma=sigma)

def moments_h_sv_ar1(omega, phi, sigma):
    return moments_h_sv_ar1_case_a(omega=omega, phi=phi, sigma=sigma)

def var_r2_kurtosis_sv_ar1(omega, phi, sigma, kappa=3.0):
    return var_r2_kurtosis_sv_ar1_case_a(omega=omega, phi=phi, sigma=sigma, kappa=kappa)

def theoretical_acf_sq_sv_ar1(nlags, omega, phi, sigma, kappa=3.0):
    return theoretical_acf_sq_sv_ar1_case_a(nlags=nlags, omega=omega, phi=phi, sigma=sigma, kappa=kappa)

def autocov_sq_sv_ar1(nlags, omega, phi, sigma, kappa=3.0):
    return autocov_sq_sv_ar1_case_a(nlags=nlags, omega=omega, phi=phi, sigma=sigma, kappa=kappa)
