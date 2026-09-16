#--------------------------------------------------------------------------
# functions for Stan support via rstan
#--------------------------------------------------------------------------

# taken from rstan:
.rstan_relist <- function(x, skeleton) {
  lst <- utils::relist(x, skeleton)
  for (i in seq_along(skeleton)) {
    dim(lst[[i]]) <- dim(skeleton[[i]])
  }
  lst
}

# taken from rstan:
.create_skeleton <- function(pars, dims) {
  lst <- lapply(seq_along(pars), function(i) {
    len_dims <- length(dims[[i]])
    if (len_dims < 1) {
      return(0)
    }
    return(array(0, dim = dims[[i]]))
  })
  names(lst) <- pars
  lst
}

.stan_log_posterior <- function(s.row, data) {
  out <- tryCatch(
    rstan::log_prob(object = data$stanfit, upars = s.row),
    error = function(e) -Inf
  )
  if (is.na(out)) {
    out <- -Inf
  }
  return(out)
}

.cmdstan_log_posterior <- function(s.row, data) {
  if ("lp__" %in% names(s.row)) {
    s.row <- s.row[!names(s.row) %in% "lp__"]
  }

  if (!is.numeric(s.row)) {
    s.row <- as.numeric(s.row)
  }
  out <- tryCatch(
    {
      log_prob <- data$log_prob(s.row, jacobian = TRUE)
      log_prob
    },
    error = function(e) {
      print(e)
      -Inf
    }
  )

  if (is.na(out)) {
    out <- -Inf
  }
  result <- data.frame(matrix(s.row, nrow = 1))
  result$log_posterior <- out

  return(out)
}

#--------------------------------------------------------------------------
# posterior scores for the score-matching proposal fit
#--------------------------------------------------------------------------

# Evaluate the posterior score s(theta) = grad log p(theta) at every row
# of `draws`, which holds draws on the unconstrained scale. Returns an
# nrow(draws) x ncol(draws) matrix, or NULL if `proposal_fit` does not
# need gradients or they could not be computed (in which case the caller
# falls back to the sample covariance). Rows whose gradient could not be
# evaluated are returned as NA and dropped, with a warning, by
# .bridge.sampler.normal().
.cmdstan_gradients <- function(samples, draws, proposal_fit) {
  if (proposal_fit == "sample") {
    return(NULL)
  }
  # grad_log_prob() is exposed by cmdstanr's init_model_methods(); call
  # it here so that users do not have to do so themselves.
  ok <- tryCatch(
    {
      suppressMessages(samples$init_model_methods())
      TRUE
    },
    error = function(e) FALSE
  )
  if (!ok) {
    warning(
      "could not initialise the cmdstanr model methods needed for ",
      "proposal_fit = 'hybrid'; falling back to the sample covariance.",
      call. = FALSE
    )
    return(NULL)
  }
  out <- tryCatch(
    {
      gm <- matrix(
        NA_real_,
        nrow = nrow(draws),
        ncol = ncol(draws),
        dimnames = list(NULL, colnames(draws))
      )
      for (i in seq_len(nrow(draws))) {
        g <- tryCatch(
          samples$grad_log_prob(unconstrained_variables = draws[i, ]),
          error = function(e) rep(NA_real_, ncol(draws))
        )
        gm[i, ] <- as.numeric(g)
      }
      gm
    },
    error = function(e) {
      warning(
        "grad_log_prob() failed (",
        conditionMessage(e),
        "); falling back to the sample covariance.",
        call. = FALSE
      )
      NULL
    }
  )
  out
}

# rstan counterpart of .cmdstan_gradients(). `upars` holds draws on the
# unconstrained scale, so adjust_transform = TRUE returns the score of
# the same density that .stan_log_posterior() evaluates.
.rstan_gradients <- function(stanfit_model, upars, proposal_fit) {
  if (proposal_fit == "sample") {
    return(NULL)
  }
  out <- tryCatch(
    {
      gm <- matrix(
        NA_real_,
        nrow = nrow(upars),
        ncol = ncol(upars),
        dimnames = list(NULL, colnames(upars))
      )
      for (i in seq_len(nrow(upars))) {
        g <- tryCatch(
          rstan::grad_log_prob(
            object = stanfit_model,
            upars = upars[i, ],
            adjust_transform = TRUE
          ),
          error = function(e) rep(NA_real_, ncol(upars))
        )
        gm[i, ] <- as.numeric(g)
      }
      gm
    },
    error = function(e) {
      warning(
        "rstan::grad_log_prob() failed (",
        conditionMessage(e),
        "); falling back to the sample covariance.",
        call. = FALSE
      )
      NULL
    }
  )
  out
}
