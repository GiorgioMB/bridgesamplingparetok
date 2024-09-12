
.bridge.sampler.realnvp <- function(
  samples_4_fit, # matrix with already transformed samples for fitting the
                 # proposal (rows are samples), colnames are "trans_x" where
                 # x is the parameter name
  samples_4_iter, # matrix with already transformed samples for the
                  # iterative scheme (rows are samples), colnames are "trans_x"
                  # where x is the parameter name
  neff, # effective sample size of samples_4_iter (i.e., already transformed samples), scalar
  log_posterior,
  ...,
  data,
  lb, ub,
  transTypes, # types of transformations (unbounded/lower/upperbounded) for the different parameters (named character vector)
  param_types, # Sample space for transformations (real, circular, simplex)
  cores,
  repetitions,
  packages,
  varlist,
  envir,
  rcppFile,
  maxiter,
  silent,
  verbose,
  r0,
  pareto_smoothing_all,
  pareto_smoothing_last,
  tol1,
  tol2,
  seed,
  return_always,
  keep_log_eval = FALSE,
  num_coupling_layers = 10,
  epochs = 50,
  learning_rate = 0.001) {

  if (is.null(neff))
    neff <- nrow(samples_4_iter)

  n_post <- nrow(samples_4_iter)
  if(verbose){
    print(str(as.numeric(samples_4_fit)))
  }
  transformed <- .transform_to_normal(samples_4_fit, num_coupling_layers = num_coupling_layers, epochs = epochs, learning_rate = learning_rate, verbose = verbose, seed = seed, return_model = TRUE)

  trained_realnvp <- transformed$model
  if(verbose){
    print(trained_realnvp)
    print(str(trained_realnvp))

  }
  realnvp_generated <- vector("list", repetitions)
  realnvp_log_jacobians <- vector("list", repetitions)
  for (i in seq_len(repetitions)) {
    latent_samples <- matrix(rnorm(n_post * ncol(samples_4_fit)), nrow = n_post)
    if(verbose){
      print(str(latent_samples))
    }
    realnvp_results <- trained_realnvp$inverse(latent_samples)
    realnvp_generated[[i]] <- realnvp_results[[1]]
    realnvp_log_jacobians[[i]] <- realnvp_results[[2]]
  }

  # Calculate q12: log density of the posterior samples under the proposal
  q12 <- apply(samples_4_iter, 1, function(x) {
    result <- trained_realnvp$inverse(x)
    log_density <- result[[2]]
    log_density
  })

  # Calculate q22: log density of the generated samples under the proposal
  q22 <- vector("list", repetitions)
  for (i in seq_len(repetitions)) {
    q22[[i]] <- apply(realnvp_generated[[i]], 1, function(x) {
      result <- trained_realnvp$inverse(x)
      log_density <- result[[2]]
      log_density
    })
  }

  # Evaluate q11: log posterior + Jacobian for the posterior samples
  q11 <- apply(samples_4_iter, 1, function(x) {
    posterior_val <- log_posterior(x, data = data, keep_log_eval = keep_log_eval, ...)
    jacobian_val <- trained_realnvp$forward(x)[[2]]
    posterior_val + jacobian_val
  })

  # Evaluate q21: log posterior + Jacobian for the generated samples
  q21 <- vector("list", repetitions)
  for (i in seq_len(repetitions)) {
    q21[[i]] <- apply(realnvp_generated[[i]], 1, function(x) {
      posterior_val <- log_posterior(x, data = data, keep_log_eval = keep_log_eval, ...)
      jacobian_val <- trained_realnvp$forward(x)[[2]]
      posterior_val + jacobian_val
    })
  }

  # Use the iterative updating scheme to compute the log marginal likelihood
  logml <- numeric(repetitions)
  niter <- numeric(repetitions)
  std_logmls <- numeric(repetitions)
  pareto_k_numi <- list()
  pareto_k_deni <- list()
  numi <- list()
  deni <- list()

  for (i in seq_len(repetitions)) {
    tmp <- .run.iterative.scheme(
      q11 = q11, 
      q12 = q12, 
      q21 = q21[[i]], 
      q22 = q22[[i]],
      r0 = r0, 
      tol = tol1, 
      method = "realnvp", 
      pareto_smoothing_all = pareto_smoothing_all,
      maxiter = maxiter, 
      silent = !verbose, 
      pareto_smoothing_last = pareto_smoothing_last, 
      verbose = verbose,
      criterion = "r", 
      neff = neff, 
      return_always = return_always
    )

    if (is.na(tmp$logml) & !is.null(tmp$r_vals)) {
      warning("logml could not be estimated within maxiter, rerunning with adjusted starting value. Estimate might be more variable than usual.", call. = FALSE)
      lr <- length(tmp$r_vals)
      r0_2 <- sqrt(tmp$r_vals[[lr - 1]] * tmp$r_vals[[lr]])
      tmp <- .run.iterative.scheme(
        q11 = q11, 
        q12 = q12, 
        q21 = q21[[i]], 
        q22 = q22[[i]], 
        r0 = r0_2, 
        tol = tol2, 
        method = "realnvp", 
        pareto_smoothing_last = pareto_smoothing_last,
        maxiter = maxiter, 
        silent = !verbose, 
        pareto_smoothing_all = pareto_smoothing_all, 
        verbose = verbose,
        criterion = "logml", 
        neff = neff, 
        return_always = return_always
      )
      tmp$niter <- maxiter + tmp$niter
    }
    logml[i] <- tmp$logml
    niter[i] <- tmp$niter
    numi[[i]] <- tmp$numi
    deni[[i]] <- tmp$deni
    std_logmls[i] <- tmp$std_logml
    if("pareto_k" %in% names(tmp)) {
      if(verbose){
        print(tmp$pareto_k)
      }
      pareto_k_numi[[i]] <- tmp$pareto_k$numi
      pareto_k_deni[[i]] <- tmp$pareto_k$deni
    } else {
      if(verbose){
        print("There was an error computing the pareto_k diagnostic")
      }
      pareto_k_numi[[i]] <- NA
      pareto_k_deni[[i]] <- NA
    }
    if (niter[i] == maxiter)
      warning("logml could not be estimated within maxiter, returning NA.", call. = FALSE)
  }
  if (repetitions == 1) {
    out <- list(logml = logml, niter = niter, method = "warp3", q11 = q11,
                q12 = q12, q21 = q21[[1]], q22 = q22[[1]], pareto_k_numi = pareto_k_numi,
                pareto_k_deni = pareto_k_deni, mcse_logml = std_logmls)
    class(out) <- "bridge"
  } else if (repetitions > 1) {
    out <- list(logml = logml, niter = niter, method = "warp3", repetitions = repetitions,
                pareto_k_numi = pareto_k_numi, pareto_k_deni = pareto_k_deni, mcse_logml = std_logmls)
    class(out) <- "bridge_list"
  }

  return(out)

}
