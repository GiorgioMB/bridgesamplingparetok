context("closed-form optimal alpha for the hybrid covariance")

# Exposed-but-internal: refer to the dotted helper directly.
.opt <- bridgesampling:::.optimal_alpha_score


# ---- A1: vectorised matches a brute-force loop reference ------------------
test_that(".optimal_alpha_score matches brute-force loop on a 50-D MVN", {
  set.seed(101)
  n <- 600L
  p <- 50L
  A <- matrix(rnorm(p * p), p, p)
  Sigma_true <- crossprod(A) / p + 0.5 * diag(p)
  Prec_true <- chol2inv(chol(Sigma_true))
  Theta <- t(replicate(n, as.numeric(chol(Sigma_true) %*% rnorm(p))))
  G <- -t(apply(Theta, 1, function(t) Prec_true %*% t))

  res <- .opt(Theta, G)

  # Brute-force reference of V_n and C_n_score using the same Sigma_n,
  # Sigma_score that the helper used internally.
  Sn <- res$Sigma_n
  Sis <- res$Sigma_score
  Tc <- sweep(Theta, 2L, colMeans(Theta))
  Sc <- sweep(G, 2L, colMeans(G))

  brute_V <- 0
  brute_C <- 0
  for (i in seq_len(n)) {
    w_i <- tcrossprod(Tc[i, ])
    T_i <- -Sis %*% tcrossprod(Sc[i, ]) %*% Sis / n
    brute_V <- brute_V + sum((w_i - Sn)^2)
    brute_C <- brute_C + sum((w_i - Sn) * (T_i - Sis))
  }
  brute_V <- brute_V / n^2
  brute_C <- brute_C / n^2

  expect_equal(res$V_n, brute_V, tolerance = 1e-9)
  expect_equal(res$C_n_score, brute_C, tolerance = 1e-9)

  expect_true(is.finite(res$alpha_star))
  expect_gte(res$alpha_star, 0)
  expect_lte(res$alpha_star, 1)
  expect_true(is.finite(res$alpha_star_scalar))
  expect_gte(res$alpha_star_scalar, 0)
  expect_lte(res$alpha_star_scalar, 1)
})


# ---- A1 edge case: nearly-degenerate target ------------------------------
test_that(".optimal_alpha_score warns and returns alpha_star = 0 when target ~ Sigma_n", {
  set.seed(202)
  p <- 4L
  n <- 1000L
  Theta <- matrix(rnorm(n * p), n, p)
  Sigma_n <- crossprod(sweep(Theta, 2L, colMeans(Theta))) / n

  # Force the helper's Sigma_score to equal Sigma_n exactly (skipping the
  # gradient-derived computation) by passing both via the API.
  G_dummy <- matrix(rnorm(n * p), n, p) # never used because both Sigmas are supplied
  expect_warning(
    res <- .opt(Theta, G_dummy, Sigma_n = Sigma_n, Sigma_score = Sigma_n),
    "essentially identical"
  )
  expect_identical(res$alpha_star, 0)
  expect_lt(res$gamma_frob, 1e-10)
})


# ---- A1 edge case: very biased target -> alpha_star at boundary ----------
test_that(".optimal_alpha_score clips alpha_star into [0, 1]", {
  set.seed(303)
  p <- 6L
  n <- 800L
  Theta <- matrix(rnorm(n * p), n, p)
  Sigma_n <- crossprod(sweep(Theta, 2L, colMeans(Theta))) / n
  bogus_target <- 1e6 * diag(p) # extremely biased

  G_dummy <- matrix(rnorm(n * p), n, p)
  res <- .opt(Theta, G_dummy, Sigma_n = Sigma_n, Sigma_score = bogus_target)
  expect_gte(res$alpha_star, 0)
  expect_lte(res$alpha_star, 1)
})


# ---- A2 + A3: bridge_sampler end-to-end on a 5-D Gaussian -----------------
.make_gaussian_problem <- function(n = 4000L, p = 5L, seed = 7) {
  set.seed(seed)
  A <- matrix(rnorm(p * p), p, p)
  Sigma_true <- crossprod(A) / p + 0.5 * diag(p)
  Prec_true <- chol2inv(chol(Sigma_true))
  Theta <- t(replicate(n, as.numeric(chol(Sigma_true) %*% rnorm(p))))
  G <- -t(apply(Theta, 1, function(t) Prec_true %*% t))
  log_density <- function(s, data, ...) -0.5 * as.numeric(t(s) %*% Prec_true %*% s)
  true_logZ <- (p / 2) * log(2 * pi) +
    0.5 * as.numeric(determinant(Sigma_true, log = TRUE)$modulus)
  list(Theta = Theta, G = G, log_density = log_density, true_logZ = true_logZ, p = p)
}

test_that("bridge_sampler with proposal_fit = 'hybrid', alpha_score = NULL works", {
  pr <- .make_gaussian_problem()
  log_density <- pr$log_density
  assign("log_density", log_density, envir = .GlobalEnv)
  on.exit(rm("log_density", envir = .GlobalEnv), add = TRUE)

  colnames(pr$Theta) <- paste0("x", seq_len(pr$p))
  lb <- rep(-Inf, pr$p)
  ub <- rep(Inf, pr$p)
  names(lb) <- names(ub) <- colnames(pr$Theta)

  out <- suppressWarnings(bridge_sampler(
    samples = pr$Theta, log_posterior = log_density,
    data = NULL, lb = lb, ub = ub, method = "normal", silent = TRUE,
    gradients = pr$G, proposal_fit = "hybrid", alpha_score = NULL
  ))

  info <- out$proposal_fit_info
  expect_equal(info$proposal_fit, "hybrid")
  expect_null(info$alpha_score_input)
  expect_true(is.finite(info$alpha_score_used))
  expect_gte(info$alpha_score_used, 0)
  expect_lte(info$alpha_score_used, 1)
  expect_equal(info$alpha_score_used, info$alpha_star)
  expect_true(is.finite(info$alpha_star_scalar))
  expect_true(is.finite(info$V_n) && info$V_n > 0)
  expect_true(is.finite(info$gamma_frob) && info$gamma_frob > 0)
  expect_true(isTRUE(info$scores_centered))

  expect_equal(out$logml, pr$true_logZ, tolerance = 0.05)
})

test_that("bridge_sampler hybrid alpha=0.5 still records optimal-alpha diagnostic", {
  pr <- .make_gaussian_problem()
  log_density <- pr$log_density
  assign("log_density", log_density, envir = .GlobalEnv)
  on.exit(rm("log_density", envir = .GlobalEnv), add = TRUE)

  colnames(pr$Theta) <- paste0("x", seq_len(pr$p))
  lb <- rep(-Inf, pr$p)
  ub <- rep(Inf, pr$p)
  names(lb) <- names(ub) <- colnames(pr$Theta)

  out <- suppressWarnings(bridge_sampler(
    samples = pr$Theta, log_posterior = log_density,
    data = NULL, lb = lb, ub = ub, method = "normal", silent = TRUE,
    gradients = pr$G, proposal_fit = "hybrid", alpha_score = 0.5
  ))

  info <- out$proposal_fit_info
  expect_equal(info$alpha_score_input, 0.5)
  expect_equal(info$alpha_score_used, 0.5)
  expect_true(is.finite(info$alpha_star))
  expect_gte(info$alpha_star, 0)
  expect_lte(info$alpha_star, 1)
})

test_that("bridge_sampler proposal_fit = 'sample' returns minimal proposal_fit_info", {
  pr <- .make_gaussian_problem(n = 2000L)
  log_density <- pr$log_density
  assign("log_density", log_density, envir = .GlobalEnv)
  on.exit(rm("log_density", envir = .GlobalEnv), add = TRUE)

  colnames(pr$Theta) <- paste0("x", seq_len(pr$p))
  lb <- rep(-Inf, pr$p)
  ub <- rep(Inf, pr$p)
  names(lb) <- names(ub) <- colnames(pr$Theta)

  out <- suppressWarnings(bridge_sampler(
    samples = pr$Theta, log_posterior = log_density,
    data = NULL, lb = lb, ub = ub, method = "normal", silent = TRUE
  ))

  info <- out$proposal_fit_info
  expect_equal(info$proposal_fit, "sample")
  expect_equal(info$alpha_score_used, 0)
  expect_true(is.na(info$alpha_star))
  expect_true(is.na(info$alpha_star_scalar))
  expect_true(is.na(info$scores_centered))
})

test_that("bridge_sampler rejects malformed alpha_score", {
  pr <- .make_gaussian_problem(n = 2000L)
  log_density <- pr$log_density
  assign("log_density", log_density, envir = .GlobalEnv)
  on.exit(rm("log_density", envir = .GlobalEnv), add = TRUE)

  colnames(pr$Theta) <- paste0("x", seq_len(pr$p))
  lb <- rep(-Inf, pr$p)
  ub <- rep(Inf, pr$p)
  names(lb) <- names(ub) <- colnames(pr$Theta)

  expect_error(
    suppressWarnings(bridge_sampler(
      samples = pr$Theta, log_posterior = log_density,
      data = NULL, lb = lb, ub = ub, method = "normal", silent = TRUE,
      gradients = pr$G, proposal_fit = "hybrid", alpha_score = 1.5
    )),
    "alpha_score"
  )
  expect_error(
    suppressWarnings(bridge_sampler(
      samples = pr$Theta, log_posterior = log_density,
      data = NULL, lb = lb, ub = ub, method = "normal", silent = TRUE,
      gradients = pr$G, proposal_fit = "hybrid", alpha_score = "auto"
    )),
    "alpha_score"
  )
})
