#--------------------------------------------------------------------------
# functions for Stan support
#--------------------------------------------------------------------------

# taken from rstan:
.rstan_relist <- function (x, skeleton) {
  lst <- utils::relist(x, skeleton)
  for (i in seq_along(skeleton)) dim(lst[[i]]) <- dim(skeleton[[i]])
  lst
}

# taken from rstan:
.create_skeleton <- function (pars, dims) {
  lst <- lapply(seq_along(pars), function(i) {
    len_dims <- length(dims[[i]])
    if (len_dims < 1)
      return(0)
    return(array(0, dim = dims[[i]]))
  })
  names(lst) <- pars
  lst
}

# The internal samplers always call log_posterior() with the extra
# `keep_log_eval` argument, which the package's own Stan helpers
# (.stan_log_posterior, .cmdstan_log_posterior) accept. User-supplied
# log_posterior functions follow the documented two-argument interface
# log_posterior(samples.row, data) and would fail with
# "unused argument (keep_log_eval = FALSE)". Wrap those so the internal
# call convention works for both; functions that take `keep_log_eval`
# or `...` are returned unchanged.
# `log_posterior` may also be given as the name of a function (the
# convention used with `varlist` / `rcppFile` for parallel runs); it is
# resolved in `envir` when possible so the same check applies, and left
# untouched when it is only defined on the workers.
.wrap_log_posterior <- function(log_posterior, envir = parent.frame()) {
  if (is.character(log_posterior)) {
    resolved <- tryCatch(get(log_posterior, envir = envir, mode = "function"),
                         error = function(e) NULL)
    if (is.null(resolved)) return(log_posterior)
    log_posterior <- resolved
  }
  if (!is.function(log_posterior)) return(log_posterior)
  fmls <- names(formals(log_posterior))
  if ("keep_log_eval" %in% fmls || "..." %in% fmls) return(log_posterior)
  function(s.row, data, keep_log_eval = FALSE, ...) log_posterior(s.row, data, ...)
}

.stan_log_posterior <- function(s.row, data, keep_log_eval) {
  out <- tryCatch(rstan::log_prob(object = data$stanfit, upars = s.row), error = function(e) -Inf)
  if (is.na(out)) out <- -Inf
  result <- data.frame(matrix(s.row, nrow = 1))
  result$log_posterior <- out
  csv_file <- "rstan_log_eval.csv"
  # Append the result to the CSV file
  if (keep_log_eval) {
    if (!file.exists(csv_file)) {
      write.csv(result, file = csv_file, row.names = FALSE)
    } else {
      write.table(result, file = csv_file, row.names = FALSE, col.names = FALSE, append = TRUE, sep = ",")
    }
  }
  return(out)
}
                  
.cmdstan_log_posterior <- function(s.row, data, keep_log_eval) {
  if("lp__" %in% names(s.row)) {
    s.row <- s.row[!names(s.row) %in% "lp__"]
    }
  #add a print that checks if fit is a CmdStanMCMC object
  if(!inherits(data, "CmdStanMCMC")) {
        stop("Passed model must be a CmdStanMCMC object")
  }
  if(!is.numeric(s.row)) {
    s.row <- as.numeric(s.row)
  }
  out <- tryCatch({
    log_prob <- data$log_prob(s.row)
    log_prob
  }, error = function(e) {
    print(e)
    -Inf
  })
  

  if (is.na(out)) {
    out <- -Inf
  }
  result <- data.frame(matrix(s.row, nrow = 1))
  result$log_posterior <- out
  csv_file <- "cmdstanr_log_eval.csv"
  # Append the result to the CSV file
  if (keep_log_eval) {
    if (!file.exists(csv_file)) {
      write.csv(result, file = csv_file, row.names = FALSE)
    } else {
      write.table(result, file = csv_file, row.names = FALSE, col.names = FALSE, append = TRUE, sep = ",")
    }
  }
  
  return(out)
}

                  
#--------------------------------------------------------------------------
# functions for Student t-Distribution
#--------------------------------------------------------------------------
                  
.estimate_df <- function(data) {
    if (!requireNamespace("moments", quietly = TRUE)) {
        stop("The 'moments' package is needed for this function to work. Please install it using install.packages('moments')")
    }

    # Calculate kurtosis for each column
    kurt_values <- apply(data, 2, moments::kurtosis)

    # Use median kurtosis to reduce the impact of outliers
    median_kurtosis <- median(kurt_values)

    kurtosis_threshold <- 100 ## Arbitrary threshold for extremely high kurtosis    
    if (median_kurtosis > kurtosis_threshold) {
        warning("Kurtosis is extremely high, suggesting df <= 3. Returning df = 3.")
        df <- 3
    } else if (median_kurtosis > 3) {
        # Apply the standard formula for df if kurtosis is above 3
        df <- 6 / (median_kurtosis - 3) + 4
    } else {
        # If kurtosis <= 3, assume the distribution is Gaussian-like
        df <- Inf
    }

    return(df)
}
