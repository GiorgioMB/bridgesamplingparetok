context("proposal_fit = 'hybrid' with Stan fits")

# A conjugate model whose log marginal likelihood is available in closed
# form. Each column of y is an independent group with its own mean, so
# the model has J unconstrained parameters and the marginal likelihood
# factorises over columns:
#
#   mu_j ~ N(0, tau),  y_ij | mu_j ~ N(mu_j, sigma)
#   => y_.j ~ N(0, sigma^2 I + tau^2 1 1')
#
# J > 1 matters here: with a single parameter the sample and
# score-matching covariance estimates are both scalars and the geometric
# mean has nothing to combine.
#
# The model uses target += ..._lpdf rather than ~, so that Stan keeps the
# normalizing constants and log_prob() is the log of the unnormalized
# posterior on the same scale as the closed form below.
hybrid_stan_code <- "
data {
  int<lower=1> J;
  int<lower=1> N;
  matrix[N, J] y;
  real<lower=0> sigma;
  real<lower=0> tau;
}
parameters {
  vector[J] mu;
}
model {
  target += normal_lpdf(mu | 0, tau);
  for (j in 1:J) {
    target += normal_lpdf(y[, j] | mu[j], sigma);
  }
}"

hybrid_stan_data <- function(J = 5L, N = 30L, sigma = 1, tau = 2, seed = 4321) {
  set.seed(seed)
  mu <- rnorm(J, 0, tau)
  y <- matrix(rnorm(N * J, rep(mu, each = N), sigma), nrow = N, ncol = J)
  list(J = J, N = N, y = y, sigma = sigma, tau = tau)
}

hybrid_stan_logml <- function(dat) {
  V <- dat$sigma^2 * diag(dat$N) + dat$tau^2 * matrix(1, dat$N, dat$N)
  sum(vapply(
    seq_len(dat$J),
    function(j) {
      mvtnorm::dmvnorm(dat$y[, j], mean = rep(0, dat$N), sigma = V, log = TRUE)
    },
    numeric(1)
  ))
}

test_that("CmdStanMCMC: proposal_fit = 'hybrid' uses the gradients", {
  testthat::skip_on_cran()
  testthat::skip_if_not_installed("cmdstanr")
  if (!file.exists(cmdstanr::cmdstan_path())) {
    testthat::skip("CmdStan is not installed in the expected path for cmdstanr.")
  }

  dat <- hybrid_stan_data()
  log_true <- hybrid_stan_logml(dat)

  tf <- tempfile(fileext = ".stan")
  on.exit(unlink(tf), add = TRUE)
  writeLines(hybrid_stan_code, tf)
  mod <- cmdstanr::cmdstan_model(tf, quiet = TRUE)

  fit <- mod$sample(
    data = dat,
    seed = 909,
    chains = 2,
    parallel_chains = 2,
    iter_warmup = 1000,
    iter_sampling = 2000,
    refresh = 0,
    show_messages = FALSE,
    show_exceptions = FALSE
  )

  bs_sample <- bridgesampling::bridge_sampler(
    fit,
    silent = TRUE,
    use_neff = FALSE
  )
  bs_hybrid <- bridgesampling::bridge_sampler(
    fit,
    silent = TRUE,
    use_neff = FALSE,
    proposal_fit = "hybrid"
  )

  # the gradients were actually obtained and used
  testthat::expect_equal(bs_hybrid$proposal_fit_info$combiner, "geometric")
  testthat::expect_equal(
    bs_hybrid$proposal_fit_info$n_gradients_used,
    2000L
  )
  testthat::expect_equal(bs_sample$proposal_fit_info$combiner, "none")

  # and the estimate is still correct
  testthat::expect_true(is.finite(bs_hybrid$logml))
  testthat::expect_lt(abs(bs_hybrid$logml - log_true), 0.1)
  testthat::expect_lt(abs(bs_hybrid$logml - bs_sample$logml), 0.1)
})

test_that("CmdStanMCMC: the default is unchanged by the new argument", {
  testthat::skip_on_cran()
  testthat::skip_if_not_installed("cmdstanr")
  if (!file.exists(cmdstanr::cmdstan_path())) {
    testthat::skip("CmdStan is not installed in the expected path for cmdstanr.")
  }

  dat <- hybrid_stan_data(J = 2L, N = 20L)

  tf <- tempfile(fileext = ".stan")
  on.exit(unlink(tf), add = TRUE)
  writeLines(hybrid_stan_code, tf)
  mod <- cmdstanr::cmdstan_model(tf, quiet = TRUE)
  fit <- mod$sample(
    data = dat,
    seed = 55,
    chains = 1,
    iter_warmup = 500,
    iter_sampling = 1000,
    refresh = 0,
    show_messages = FALSE,
    show_exceptions = FALSE
  )

  set.seed(1)
  a <- bridgesampling::bridge_sampler(fit, silent = TRUE, use_neff = FALSE)
  set.seed(1)
  b <- bridgesampling::bridge_sampler(
    fit,
    silent = TRUE,
    use_neff = FALSE,
    proposal_fit = "sample"
  )
  testthat::expect_equal(a$logml, b$logml)
  testthat::expect_equal(a$proposal_fit_info$combiner, "none")
})

test_that("stanfit: proposal_fit = 'hybrid' uses the gradients", {
  testthat::skip_on_cran()
  testthat::skip_on_os("windows")
  testthat::skip_if_not_installed("rstan")

  dat <- hybrid_stan_data()
  log_true <- hybrid_stan_logml(dat)

  sm <- suppressWarnings(
    rstan::stan_model(model_code = hybrid_stan_code, model_name = "hybrid_test")
  )
  fit <- rstan::sampling(
    sm,
    data = dat,
    seed = 909,
    chains = 2,
    iter = 3000,
    warmup = 1000,
    refresh = 0
  )

  bs_sample <- bridgesampling::bridge_sampler(
    fit,
    silent = TRUE,
    use_neff = FALSE
  )
  bs_hybrid <- bridgesampling::bridge_sampler(
    fit,
    silent = TRUE,
    use_neff = FALSE,
    proposal_fit = "hybrid"
  )

  testthat::expect_equal(bs_hybrid$proposal_fit_info$combiner, "geometric")
  testthat::expect_equal(bs_sample$proposal_fit_info$combiner, "none")
  testthat::expect_true(is.finite(bs_hybrid$logml))
  testthat::expect_lt(abs(bs_hybrid$logml - log_true), 0.1)
  testthat::expect_lt(abs(bs_hybrid$logml - bs_sample$logml), 0.1)
})
