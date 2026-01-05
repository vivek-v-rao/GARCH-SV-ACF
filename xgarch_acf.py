# xgarch_acf.py

import numpy as np

from garch_acf import (
    simulate_garch_1_1,
    simulate_nagarch_1_1,
    simulate_gjr_garch_1_1,
    simulate_figarch_1_1,
    acf,
    crosscorr_lead,
    theoretical_acf_sq_garch_1_1,
    theoretical_acf_sq_nagarch_1_1,
    theoretical_acf_sq_gjr_garch_1_1,
    theoretical_acf_sq_figarch_1_1,
    theoretical_crosscorr_r_r2_nagarch_1_1,
    theoretical_crosscorr_r_r2_gjr_garch_1_1,
    acf_abs_garch_1_1,
    autocov_abs_garch_1_1,
    acf_abs_nagarch_1_1,
    autocov_abs_nagarch_1_1,
    acf_abs_gjr_garch_1_1,
    autocov_abs_gjr_garch_1_1,
    acf_abs_figarch_1_1,
    autocov_abs_figarch_1_1,
    simulate_egarch_1_1,
    theoretical_acf_sq_egarch_1_1,
    autocov_sq_egarch_1_1,
    autocov_abs_egarch_1_1,
    acf_abs_egarch_1_1,
    theoretical_crosscorr_r_r2_egarch_1_1,
    simulate_sv_ar1_case_a,
    theoretical_acf_sq_sv_ar1_case_a,
    autocov_sq_sv_ar1_case_a
)


def autocov(x, nlags):
    """
    Sample autocovariance for lags 0..nlags:

        out[k] = (1/(n-k)) * sum_{t=k..n-1} (x_t-m)(x_{t-k}-m)
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    m = x.mean()
    x0 = x - m

    out = np.empty(nlags + 1, dtype=float)
    for k in range(0, nlags + 1):
        out[k] = np.dot(x0[k:], x0[: n - k]) / (n - k)
    return out


def print_compare_acf(title, emp, th, nlags):
    print(title)
    print("%3s  %14s  %14s  %14s" % ("lag", "empirical", "theoretical", "emp-th"))
    for k in range(1, nlags + 1):
        print("%3d  %14.8f  %14.8f  %14.8f" % (k, emp[k], th[k], emp[k] - th[k]))
    print()


def print_compare_autocov(title, emp, th, nlags):
    print(title)
    print("%3s  %14s  %14s  %14s" % ("lag", "empirical", "theoretical", "emp-th"))
    for k in range(0, nlags + 1):
        print("%3d  %14.8f  %14.8f  %14.8f" % (k, emp[k], th[k], emp[k] - th[k]))
    print()


def main():
    # -------------------------
    # symmetric garch(1,1)
    # -------------------------
    omega = 0.1
    alpha = 0.10
    beta = 0.85

    n = 200_000
    nlags = 20
    burn = 10_000
    seed = 1

    eps, _h = simulate_garch_1_1(n, omega, alpha, beta, burn=burn, seed=seed)

    emp_acf_sq = acf(eps ** 2, nlags)
    th_acf_sq = theoretical_acf_sq_garch_1_1(alpha, beta, nlags)

    print("symmetric garch(1,1)")
    print("omega=%.6g alpha=%.6g beta=%.6g" % (omega, alpha, beta))
    print("n=%d nlags=%d burn=%d seed=%d" % (n, nlags, burn, seed))
    print()
    print("acf of squared returns")
    print("%3s  %14s  %14s  %14s" % ("lag", "empirical", "theoretical", "emp-th"))
    for k in range(1, nlags + 1):
        print("%3d  %14.8f  %14.8f  %14.8f" % (k, emp_acf_sq[k], th_acf_sq[k], emp_acf_sq[k] - th_acf_sq[k]))
    print()

    # tier a tests for |eps|
    kappa = 3.0
    abs_eps = np.abs(eps)

    emp_acf_abs = acf(abs_eps, nlags)
    th_acf_abs, th_var_abs, th_mean_abs = acf_abs_garch_1_1(omega, alpha, beta, kappa, nlags)

    print("acf of absolute returns (tier a)")
    print(
        "kappa=%.6g  mean(|eps|): emp=%.8g th=%.8g  var(|eps|): emp=%.8g th=%.8g"
        % (kappa, float(np.mean(abs_eps)), th_mean_abs, float(np.var(abs_eps)), th_var_abs)
    )
    print()
    print_compare_acf("acf(|eps|)", emp_acf_abs, th_acf_abs, nlags)

    emp_acv_abs = autocov(abs_eps, nlags)
    th_acv_abs, th_var_abs2, th_mean_abs2 = autocov_abs_garch_1_1(omega, alpha, beta, kappa, nlags)

    print("autocov of absolute returns (tier a)")
    print(
        "kappa=%.6g  mean(|eps|): emp=%.8g th=%.8g  var(|eps|): emp=%.8g th=%.8g"
        % (kappa, float(np.mean(abs_eps)), th_mean_abs2, float(emp_acv_abs[0]), th_var_abs2)
    )
    print()
    print_compare_autocov("autocov(|eps|)", emp_acv_abs, th_acv_abs, nlags)

    # -------------------------
    # nagarch(1,1) "2b"
    # -------------------------
    omega = 0.1
    alpha = 0.10
    beta = 0.85
    theta = 0.20

    n = 2_000_000
    nlags = 20
    burn = 50_000
    seed = 2

    eps, h = simulate_nagarch_1_1(n, omega, alpha, beta, theta, burn=burn, seed=seed)

    emp_acf_sq = acf(eps ** 2, nlags)
    th_acf_sq = theoretical_acf_sq_nagarch_1_1(alpha, beta, theta, nlags)

    print("nagarch(1,1) (2b form)")
    print("omega=%.6g alpha=%.6g beta=%.6g theta=%.6g" % (omega, alpha, beta, theta))
    print("n=%d nlags=%d burn=%d seed=%d" % (n, nlags, burn, seed))
    print()
    print("acf of squared returns")
    print("%3s  %14s  %14s  %14s" % ("lag", "empirical", "theoretical", "emp-th"))
    for k in range(1, nlags + 1):
        print("%3d  %14.8f  %14.8f  %14.8f" % (k, emp_acf_sq[k], th_acf_sq[k], emp_acf_sq[k] - th_acf_sq[k]))
    print()

    emp_cc = crosscorr_lead(eps, eps ** 2, nlags)

    eh32 = float(np.mean(h ** 1.5))
    th_cc = theoretical_crosscorr_r_r2_nagarch_1_1(omega, alpha, beta, theta, nlags, eh32=eh32)

    print("crosscorr: corr(eps_t, eps_{t+k}^2)")
    print("note: theoretical uses eh32 = mean(h**1.5) from this run (eh32=%.6g)" % eh32)
    print("%3s  %14s  %14s  %14s" % ("lag", "empirical", "theoretical", "emp-th"))
    for k in range(1, nlags + 1):
        print("%3d  %14.8f  %14.8f  %14.8f" % (k, emp_cc[k], th_cc[k], emp_cc[k] - th_cc[k]))
    print()

    # tier a tests for |eps|
    kappa = 3.0
    abs_eps = np.abs(eps)

    emp_acf_abs = acf(abs_eps, nlags)
    th_acf_abs, th_var_abs, th_mean_abs = acf_abs_nagarch_1_1(omega, alpha, beta, theta, kappa, nlags)

    print("acf of absolute returns (tier a)")
    print(
        "kappa=%.6g  mean(|eps|): emp=%.8g th=%.8g  var(|eps|): emp=%.8g th=%.8g"
        % (kappa, float(np.mean(abs_eps)), th_mean_abs, float(np.var(abs_eps)), th_var_abs)
    )
    print()
    print_compare_acf("acf(|eps|)", emp_acf_abs, th_acf_abs, nlags)

    emp_acv_abs = autocov(abs_eps, nlags)
    th_acv_abs, th_var_abs2, th_mean_abs2 = autocov_abs_nagarch_1_1(omega, alpha, beta, theta, kappa, nlags)

    print("autocov of absolute returns (tier a)")
    print(
        "kappa=%.6g  mean(|eps|): emp=%.8g th=%.8g  var(|eps|): emp=%.8g th=%.8g"
        % (kappa, float(np.mean(abs_eps)), th_mean_abs2, float(emp_acv_abs[0]), th_var_abs2)
    )
    print()
    print_compare_autocov("autocov(|eps|)", emp_acv_abs, th_acv_abs, nlags)

    # -------------------------
    # gjr-garch(1,1)
    # -------------------------
    omega = 0.1
    alpha = 0.06
    beta = 0.85
    gamma = 0.08

    n = 2_000_000
    nlags = 20
    burn = 50_000
    seed = 3

    eps, _h = simulate_gjr_garch_1_1(n, omega, alpha, beta, gamma, burn=burn, seed=seed)

    emp_acf_sq = acf(eps ** 2, nlags)
    th_acf_sq = theoretical_acf_sq_gjr_garch_1_1(alpha, beta, gamma, nlags)

    print("gjr-garch(1,1)")
    print("omega=%.6g alpha=%.6g beta=%.6g gamma=%.6g" % (omega, alpha, beta, gamma))
    print("n=%d nlags=%d burn=%d seed=%d" % (n, nlags, burn, seed))
    print()
    print("acf of squared returns")
    print("%3s  %14s  %14s  %14s" % ("lag", "empirical", "theoretical", "emp-th"))
    for k in range(1, nlags + 1):
        print("%3d  %14.8f  %14.8f  %14.8f" % (k, emp_acf_sq[k], th_acf_sq[k], emp_acf_sq[k] - th_acf_sq[k]))
    print()

    emp_cc = crosscorr_lead(eps, eps ** 2, nlags)

    method = "taylor2"
    th_cc = theoretical_crosscorr_r_r2_gjr_garch_1_1(omega, alpha, beta, gamma, nlags, method=method)

    print("crosscorr: corr(eps_t, eps_{t+k}^2)")
    print('note: theoretical uses moment approximation for E[h^(3/2)] (method="%s")' % method)
    print("%3s  %14s  %14s  %14s" % ("lag", "empirical", "theoretical", "emp-th"))
    for k in range(1, nlags + 1):
        print("%3d  %14.8f  %14.8f  %14.8f" % (k, emp_cc[k], th_cc[k], emp_cc[k] - th_cc[k]))
    print()

    # tier a tests for |eps|
    kappa = 3.0
    abs_eps = np.abs(eps)

    emp_acf_abs = acf(abs_eps, nlags)
    th_acf_abs, th_var_abs, th_mean_abs = acf_abs_gjr_garch_1_1(omega, alpha, beta, gamma, kappa, nlags)

    print("acf of absolute returns (tier a)")
    print(
        "kappa=%.6g  mean(|eps|): emp=%.8g th=%.8g  var(|eps|): emp=%.8g th=%.8g"
        % (kappa, float(np.mean(abs_eps)), th_mean_abs, float(np.var(abs_eps)), th_var_abs)
    )
    print()
    print_compare_acf("acf(|eps|)", emp_acf_abs, th_acf_abs, nlags)

    emp_acv_abs = autocov(abs_eps, nlags)
    th_acv_abs, th_var_abs2, th_mean_abs2 = autocov_abs_gjr_garch_1_1(omega, alpha, beta, gamma, kappa, nlags)

    print("autocov of absolute returns (tier a)")
    print(
        "kappa=%.6g  mean(|eps|): emp=%.8g th=%.8g  var(|eps|): emp=%.8g th=%.8g"
        % (kappa, float(np.mean(abs_eps)), th_mean_abs2, float(emp_acv_abs[0]), th_var_abs2)
    )
    print()
    print_compare_autocov("autocov(|eps|)", emp_acv_abs, th_acv_abs, nlags)

    # -------------------------
    # figarch(1,d,1)
    # -------------------------
    omega = 0.1
    alpha = 0.05
    beta = 0.50
    d = 0.30

    n = 300_000
    nlags = 20
    burn = 50_000
    seed = 4

    trunc = 500
    n_terms = 200_000
    kappa = 3.0

    eps, _h = simulate_figarch_1_1(n, omega, alpha, beta, d, trunc=trunc, burn=burn, seed=seed)

    print("figarch(1,d,1)")
    print("omega=%.6g alpha=%.6g beta=%.6g d=%.6g" % (omega, alpha, beta, d))
    print("n=%d nlags=%d burn=%d seed=%d trunc=%d n_terms=%d" % (n, nlags, burn, seed, trunc, n_terms))
    print()

    emp_acf_sq = acf(eps ** 2, nlags)
    th_acf_sq = theoretical_acf_sq_figarch_1_1(alpha, beta, d, nlags, n_terms=n_terms)

    print("acf of squared returns (arfima-based, mean-corrected)")
    print_compare_acf("acf(eps^2)", emp_acf_sq, th_acf_sq, nlags)

    abs_eps = np.abs(eps)

    emp_acf_abs = acf(abs_eps, nlags)
    th_acf_abs, th_var_abs, th_mean_abs = acf_abs_figarch_1_1(
        omega, alpha, beta, d, kappa, nlags, trunc=trunc, n_terms=n_terms
    )

    print("acf of absolute returns (tier a, delta/lognormal mapping)")
    print(
        "kappa=%.6g  mean(|eps|): emp=%.8g th=%.8g  var(|eps|): emp=%.8g th=%.8g"
        % (kappa, float(np.mean(abs_eps)), th_mean_abs, float(np.var(abs_eps)), th_var_abs)
    )
    print()
    print_compare_acf("acf(|eps|)", emp_acf_abs, th_acf_abs, nlags)

    emp_acv_abs = autocov(abs_eps, nlags)
    th_acv_abs, th_var_abs2, th_mean_abs2 = autocov_abs_figarch_1_1(
        omega, alpha, beta, d, kappa, nlags, trunc=trunc, n_terms=n_terms
    )

    print("autocov of absolute returns (tier a, delta/lognormal mapping)")
    print(
        "kappa=%.6g  mean(|eps|): emp=%.8g th=%.8g  var(|eps|): emp=%.8g th=%.8g"
        % (kappa, float(np.mean(abs_eps)), th_mean_abs2, float(emp_acv_abs[0]), th_var_abs2)
    )
    print()
    print_compare_autocov("autocov(|eps|)", emp_acv_abs, th_acv_abs, nlags)
    # -------------------------
    # egarch(1,1)
    # -------------------------
    omega = 0.0
    alpha = 0.20
    beta = 0.95
    gamma = -0.10

    n = 2_000_000
    nlags = 20
    burn = 50_000
    seed = 4

    # normal innovations => kappa = E[z^4] = 3
    kappa = 3.0

    eps, h = simulate_egarch_1_1(n, omega, alpha, beta, gamma, burn=burn, seed=seed)

    # ---- squared returns: acf + autocov/var/kurt
    emp_acf_sq = acf(eps ** 2, nlags)
    th_acf_sq = theoretical_acf_sq_egarch_1_1(alpha, beta, gamma, kappa, nlags)

    print()
    print("egarch(1,1)")
    print("omega=%.6g alpha=%.6g beta=%.6g gamma=%.6g" % (omega, alpha, beta, gamma))
    print("n=%d nlags=%d burn=%d seed=%d" % (n, nlags, burn, seed))
    print()
    print("acf of squared returns")
    print("%3s  %14s  %14s  %14s" % ("lag", "empirical", "theoretical", "emp-th"))
    for k in range(1, nlags + 1):
        print("%3d  %14.8f  %14.8f  %14.8f" % (k, emp_acf_sq[k], th_acf_sq[k], emp_acf_sq[k] - th_acf_sq[k]))

    # empirical autocov for eps^2
    eps2 = eps ** 2
    eps2c = eps2 - eps2.mean()
    emp_autocov_sq = np.empty(nlags + 1, dtype=float)
    emp_autocov_sq[0] = float(np.dot(eps2c, eps2c)) / float(eps2c.shape[0])
    for k in range(1, nlags + 1):
        emp_autocov_sq[k] = float(np.dot(eps2c[k:], eps2c[:-k])) / float(eps2c.shape[0])

    th_autocov_sq, th_var_r2, th_kurt, th_mu_h, th_m2 = autocov_sq_egarch_1_1(
        omega, alpha, beta, gamma, kappa, nlags
    )

    emp_var_r2 = float(np.var(eps2, ddof=0))
    emp_kurt = float(np.mean(eps ** 4) / (np.mean(eps ** 2) ** 2))

    print()
    print("autocov of squared returns (tier a)")
    print("kappa=%.6g  var(eps^2): emp=%.8g th=%.8g  kurt: emp=%.8g th=%.8g" % (kappa, emp_var_r2, th_var_r2, emp_kurt, th_kurt))
    print()
    print("autocov(eps^2)")
    print("%3s  %14s  %14s  %14s" % ("lag", "empirical", "theoretical", "emp-th"))
    for k in range(0, nlags + 1):
        print("%3d  %14.8f  %14.8f  %14.8f" % (k, emp_autocov_sq[k], th_autocov_sq[k], emp_autocov_sq[k] - th_autocov_sq[k]))

    # ---- crosscorr: corr(eps_t, eps_{t+k}^2)
    emp_cc = crosscorr_lead(eps, eps ** 2, nlags)
    n_terms = 1200
    th_cc = theoretical_crosscorr_r_r2_egarch_1_1(
        omega, alpha, beta, gamma, nlags, kappa=kappa, n_terms=n_terms
    )

    print()
    print("crosscorr: corr(eps_t, eps_{t+k}^2)")
    print("note: theoretical uses truncated products (n_terms=%d)" % n_terms)
    print("%3s  %14s  %14s  %14s" % ("lag", "empirical", "theoretical", "emp-th"))
    for k in range(1, nlags + 1):
        print("%3d  %14.8f  %14.8f  %14.8f" % (k, emp_cc[k], th_cc[k], emp_cc[k] - th_cc[k]))

    # ---- absolute returns: acf + autocov
    abs_eps = np.abs(eps)

    emp_mean_abs = float(np.mean(abs_eps))
    emp_var_abs = float(np.var(abs_eps, ddof=0))

    th_rho_abs, th_var_abs, th_mean_abs = acf_abs_egarch_1_1(omega, alpha, beta, gamma, nlags)

    print()
    print("acf of absolute returns (tier a/lognormal)")
    print("mean(|eps|): emp=%.8g th=%.8g  var(|eps|): emp=%.8g th=%.8g" % (emp_mean_abs, th_mean_abs, emp_var_abs, th_var_abs))
    print()
    print("acf(|eps|)")
    print("%3s  %14s  %14s  %14s" % ("lag", "empirical", "theoretical", "emp-th"))

    emp_acf_abs = acf(abs_eps, nlags)
    for k in range(1, nlags + 1):
        print("%3d  %14.8f  %14.8f  %14.8f" % (k, emp_acf_abs[k], th_rho_abs[k], emp_acf_abs[k] - th_rho_abs[k]))

    # empirical autocov for |eps|
    absc = abs_eps - abs_eps.mean()
    emp_autocov_abs = np.empty(nlags + 1, dtype=float)
    emp_autocov_abs[0] = float(np.dot(absc, absc)) / float(absc.shape[0])
    for k in range(1, nlags + 1):
        emp_autocov_abs[k] = float(np.dot(absc[k:], absc[:-k])) / float(absc.shape[0])

    th_autocov_abs, th_var_abs2, th_mean_abs2 = autocov_abs_egarch_1_1(omega, alpha, beta, gamma, nlags)

    print()
    print("autocov of absolute returns (tier a/lognormal)")
    print("mean(|eps|): emp=%.8g th=%.8g  var(|eps|): emp=%.8g th=%.8g" % (emp_mean_abs, th_mean_abs2, emp_var_abs, th_var_abs2))
    print()
    print("autocov(|eps|)")
    print("%3s  %14s  %14s  %14s" % ("lag", "empirical", "theoretical", "emp-th"))
    for k in range(0, nlags + 1):
        print("%3d  %14.8f  %14.8f  %14.8f" % (k, emp_autocov_abs[k], th_autocov_abs[k], emp_autocov_abs[k] - th_autocov_abs[k]))

    # ----------------------------------------------------------------------
    # sv log-ar(1) (case a, no leverage) - test block
    # paste this inside main() (and keep the helper funcs either above main()
    # or paste them right above this block).
    # ----------------------------------------------------------------------

    def _acf_biased(x, nlags):
        x = np.asarray(x, dtype=float)
        x = x - x.mean()
        v = np.mean(x * x)
        out = np.empty(nlags, dtype=float)
        for k in range(1, nlags + 1):
            out[k - 1] = np.mean(x[:-k] * x[k:]) / v
        return out

    def _autocov_biased(x, nlags):
        x = np.asarray(x, dtype=float)
        x0 = x - x.mean()
        out = np.empty(nlags + 1, dtype=float)
        out[0] = np.mean(x0 * x0)
        for k in range(1, nlags + 1):
            out[k] = np.mean(x0[:-k] * x0[k:])
        return out

    # parameters (pick whatever you want)
    mu = 0.0
    phi = 0.97
    sigma_eta = 0.20
    kappa = 3.0

    n = 300000
    nlags = 20
    burn = 50000
    seed = 5

    # map to the parameterization used in garch_acf.py:
    # x_t = omega + phi x_{t-1} + sigma eta_t  has mean mu = omega/(1-phi)
    omega = (1.0 - phi) * mu
    sigma = sigma_eta

    print("\nsv log-ar(1) (case a, no leverage)")
    print(f"mu={mu} phi={phi} sigma_eta={sigma_eta}")
    print(f"(mapped) omega={omega} sigma={sigma} kappa={kappa}")
    print(f"n={n} nlags={nlags} burn={burn} seed={seed}")

    eps, h = simulate_sv_ar1_case_a(
        n=n, omega=omega, phi=phi, sigma=sigma, burn=burn, seed=seed
    )

    r2 = eps * eps

    # --- acf(eps^2)
    emp_acf_r2 = _acf_biased(r2, nlags)
    th_acf_r2 = theoretical_acf_sq_sv_ar1_case_a(
        nlags=nlags, omega=omega, phi=phi, sigma=sigma, kappa=kappa
    )

    print("\nacf of squared returns")
    print("lag       empirical     theoretical          emp-th")
    for k in range(1, nlags + 1):
        e = emp_acf_r2[k - 1]
        t = th_acf_r2[k - 1]
        print(f"{k:3d} {e:14.8f} {t:14.8f} {e - t:14.8f}")

    # --- autocov(eps^2) + var + kurt
    emp_acv_r2 = _autocov_biased(r2, nlags)
    th_acv_r2 = autocov_sq_sv_ar1_case_a(
        nlags=nlags, omega=omega, phi=phi, sigma=sigma, kappa=kappa
    )

    # implied unconditional moments
    mean_x = omega / (1.0 - phi)
    var_x = (sigma * sigma) / (1.0 - phi * phi) if sigma != 0.0 else 0.0
    eh = float(np.exp(mean_x + 0.5 * var_x))              # E[h]
    eh2 = float(np.exp(2.0 * mean_x + 2.0 * var_x))       # E[h^2]

    th_var_r2 = kappa * eh2 - eh * eh
    th_kurt = (kappa * eh2) / (eh * eh)

    emp_var_r2 = float(emp_acv_r2[0])
    emp_kurt = float(np.mean(eps**4) / (np.mean(eps**2) ** 2))

    print("\nautocov of squared returns")
    print(
        f"kappa={kappa:g}  var(eps^2): emp={emp_var_r2:.7g} th={th_var_r2:.7g}  "
        f"kurt: emp={emp_kurt:.7g} th={th_kurt:.7g}"
    )
    print(f"E[h]: th={eh:.7g}  E[h^2]: th={eh2:.7g}")

    print("\nautocov(eps^2)")
    print("lag       empirical     theoretical          emp-th")
    for k in range(0, nlags + 1):
        e = emp_acv_r2[k]
        t = th_acv_r2[k]
        print(f"{k:3d} {e:14.8f} {t:14.8f} {e - t:14.8f}")

if __name__ == "__main__":
    main()
