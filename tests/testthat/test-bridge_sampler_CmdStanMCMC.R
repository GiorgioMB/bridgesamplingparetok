context('test bridge_sampler cmdstanmcmc method')

testthat::test_that("bridge_sampler() works for CmdStanMCMC and matches analytic evidence (Normal-Normal, known sigma)", {
  testthat::skip_on_cran()
  testthat::skip_if_not_installed("cmdstanr")
  testthat::skip_if_not_installed("bridgesampling")
  testthat::skip_if_not_installed("posterior")

  # Require a working CmdStan toolchain
  if (!file.exists(cmdstanr::cmdstan_path())) {
    testthat::skip("CmdStan is not installed in the expected path for cmdstanr.")
  }

  set.seed(123)

  # Simple conjugate model: y_i ~ N(mu, sigma^2), mu ~ N(mu0, tau0^2), sigma known
  N     <- 60L
  mu0   <- 0
  tau0  <- 1
  sigma <- 1
  y     <- rnorm(N, mean = 0.5, sd = sigma)

  data_list <- list(N = N, y = y, sigma = sigma)

  stan_code <- "
  data {
    int<lower=1> N;
    vector[N] y;
    real<lower=0> sigma;
  }
  parameters {
    real mu;
  }
  model {
    mu ~ normal(0, 1);          // prior mu0 = 0, tau0 = 1
    y ~ normal(mu, sigma);      // known sigma
  }
  "

  # Compile a tiny model
  tf <- withr::local_tempfile(fileext = ".stan")
  writeLines(stan_code, tf)
  mod <- cmdstanr::cmdstan_model(tf, quiet = TRUE)

  fit <- mod$sample(
    data = data_list,
    seed = 202,
    chains = 2,
    parallel_chains = 2,
    iter_warmup = 500,
    iter_sampling = 1000,
    refresh = 0
  )

  # Bridge sampling using the S3 method for CmdStanMCMC
  bs <- bridgesampling::bridge_sampler(fit, data = data_list, silent = TRUE)
  testthat::expect_s3_class(fit, "CmdStanMCMC")
  testthat::expect_true(is.list(bs))
  testthat::expect_true(is.finite(bs$logml))

  # ---- Analytic log marginal likelihood for Normal-Normal with known sigma ----
  # y ~ MVN(mu0 * 1_N, Sigma), with Sigma = sigma^2 I_N + tau0^2 J (J = 11^T)
  # Use matrix determinant lemma / Sherman-Morrison:
  # |Sigma| = (sigma^2)^(N-1) * (sigma^2 + N*tau0^2)
  # Sigma^{-1} = (1/sigma^2)I - (tau0^2 / (sigma^2*(sigma^2 + N*tau0^2))) * J
  #
  # log p(y) = -0.5 * [ N*log(2*pi) + log|Sigma| + (y - mu0)^T Sigma^{-1} (y - mu0) ]
  #
  y_centered <- y - mu0
  s2 <- sigma^2
  t2 <- tau0^2
  logdet_Sigma <- (N - 1) * log(s2) + log(s2 + N * t2)

  sum_y <- sum(y_centered)
  quad <- (1 / s2) * sum(y_centered^2) -
    (t2 / (s2 * (s2 + N * t2))) * (sum_y^2)

  logml_analytic <- -0.5 * (N * log(2 * pi) + logdet_Sigma + quad)

  # The bridge sampler estimate should be close to the analytic value.
  # Tolerance depends on MCMC & bridge variance, 0.5 should be reasonable.
  testthat::expect_equal(bs$logml, as.numeric(logml_analytic), tolerance = 0.5)

  # Just sanity: should still be finite and typically closer (not enforced strictly).
  fit2 <- mod$sample(
    data = data_list,
    seed = 203,
    chains = 4,
    parallel_chains = 4,
    iter_warmup = 750,
    iter_sampling = 2000,
    refresh = 0
  )
  bs2 <- bridgesampling::bridge_sampler(fit2, data = data_list, silent = TRUE)
  testthat::expect_true(is.finite(bs2$logml))
})
