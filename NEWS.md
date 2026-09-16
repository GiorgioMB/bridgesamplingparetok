# bridgesampling 1.2-2 (2026-08-12)

* `bridge_sampler()` with `proposal_fit = "hybrid"` now accepts `alpha_score = NULL`, triggering the closed-form Ledoit-Wolf optimal mixing weight $a^\ast = (V_n - C_{n,{\rm score}}) / \gamma_{\rm frob}$ to be computed per fit (Ledoit & Wolf 2004). The chosen alpha and its plug-in breakdown (`alpha_star`, `alpha_star_scalar`, `V_n`, `C_n_score`, `gamma_frob`) are returned in `out$proposal_fit_info` on every gradient-using call, offering the optimal value as a free diagnostic for runs using a fixed alpha.
* The score-matching covariance estimator utilizes centred scores $\Sigma_{\rm score}^{-1} = n^{-1} \sum_i (s_i - \bar s)(s_i - \bar s)^\top$, removing the $\mathcal O(\bar{s} \bar{s}^\top)$ finite-sample bias present in the previous estimator (despite $\mathbb E_\pi[s] = 0$ in the population) and implements the score-matching estimator of Hyvärinen (2005). The effect on `logml` is $\mathcal O(1/n)$ for typical posteriors.
# bridgesampling 1.2-1 (2025-11-18)

* Added CmdStanR method and corresponding tests (thanks to @GiorgioMB and @avehtari #44).
* Added Monte Carlo Standard Error (MCSE) to bridgesampling, see: https://arxiv.org/abs/2508.14487 (thanks to @GiorgioMB and @avehtari #43).
* Fixed bug in simplex with small dimensionality (thanks to @FBartos #31).
* Added a `NEWS.md` file to track changes to the package.

# bridgesampling 1.1-5 (2023-06-01)

* Deactivated stanreg tests to avoid CRAN check issues.

# bridgesampling 1.1-0 (2021-03-01)

* Fixed subscript out of bounds error, see: https://github.com/quentingronau/bridgesampling/issues/26
* Deactivated stan tests on Windows to avoid CRAN check issues.

# bridgesampling 1.0-0 (2020-02-01)

* Included citation file and references to JSS article

# bridgesampling 0.8-0 (2019-12-01)

* Disabled use of mvnfast and revetred back to mvtnorn. see also: https://github.com/quentingronau/bridgesampling/issues/20
* Version 0.7-x introduced a bug that prevented a rerunning of the iterative scheme based on harmonic mean in case maxit was reached. This bug should now be removed. See: https://github.com/quentingronau/bridgesampling/issues/18
* For older news see NEWS.old file on GitHub: https://github.com/quentingronau/bridgesampling/blob/master/NEWS.old

