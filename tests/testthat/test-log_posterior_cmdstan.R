context(".cmdstan_log_posterior helper function for CmdStanMCMC objects")
testthat::test_that(".cmdstan_log_posterior matches lp__ from CmdStanMCMC and handles inputs", {
  testthat::skip_on_cran()
  testthat::skip_if_not_installed("cmdstanr")
  testthat::skip_if_not_installed("posterior")

  pkg <- "bridgesampling"
  if (!requireNamespace(pkg, quietly = TRUE)) {
    testthat::skip("Package 'bridgesampling' not installed.")
  }
  if (!exists(".cmdstan_log_posterior", envir = asNamespace(pkg), inherits = FALSE)) {
    testthat::skip("Internal function .cmdstan_log_posterior not found in 'bridgesampling'.")
  }
  .cmdstan_log_posterior <- get(".cmdstan_log_posterior", envir = asNamespace(pkg))
  if (!file.exists(cmdstanr::cmdstan_path())) {
    testthat::skip("CmdStan is not installed in the expected path for cmdstanr.")
  }
  if (isTRUE(utils::packageVersion("cmdstanr") >= "0.7.0")) {
    ok <- tryCatch({
      invisible(cmdstanr::check_cmdstan_toolchain(fix = FALSE))
      TRUE
    }, error = function(e) FALSE)
    if (!ok) testthat::skip("CmdStan toolchain check failed on this machine.")
  }

  set.seed(321)
  N     <- 40L
  sigma <- 1
  y     <- rnorm(N, 0.25, sigma)
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
    mu ~ normal(0, 1);
    y ~ normal(mu, sigma);
  }
  generated quantities {
    real lp__copy = target();  // read current log density to ensure target is defined
  }
  "

  tf <- withr::local_tempfile(fileext = ".stan")
  writeLines(stan_code, tf)

  # Compile with graceful skip on failure (collect the real compiler error)
  mod <-  cmdstanr::cmdstan_model(tf, quiet = TRUE, force_recompile = TRUE)

  # Sample with graceful skip on failure
  fit <-mod$sample(
      data = data_list,
      seed = 404,
      chains = 2,
      parallel_chains = 2,
      iter_warmup = 300,
      iter_sampling = 800)

  # Extract CmdStan's lp__
  draws_df <- fit$draws(variables = "lp__", format = "df")
  testthat::expect_true(nrow(draws_df) > 0)

  # Compute via internal helper
  lp_vec <- .cmdstan_log_posterior(fit = fit, data = data_list)

  testthat::expect_type(lp_vec, "double")
  testthat::expect_length(lp_vec, nrow(draws_df))

  # They should match up to numerical noise
  testthat::expect_equal(lp_vec, draws_df$lp__, tolerance = 1e-8)

  # Basic input validation
  testthat::expect_error(.cmdstan_log_posterior(fit = "not-a-fit", data = data_list))
  testthat::expect_error(.cmdstan_log_posterior(fit = fit, data = list(N = N)))
})
