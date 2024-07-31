
#--------------------------------------------------------------------------
# functions for Stan support via rstan
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

.stan_log_posterior <- function(s.row, data) {
  out <- tryCatch(rstan::log_prob(object = data$stanfit, upars = s.row), error = function(e) -Inf)
  if (is.na(out)) out <- -Inf
  return(out)
}
                  
.cmdstan_log_posterior <- function(params, fit) {
    ##If params is not numeric, attempt to convert it to numeric
  if(!is.numeric(params)) {
    params <- as.numeric(params)
  }
  out <- tryCatch({
    log_prob <- fit$log_prob(params, jacobian = TRUE)
    log_prob
  }, error = function(e) {
    -Inf
  })

  if (is.na(out)) {
    out <- -Inf
  }
  
  return(out)
}
