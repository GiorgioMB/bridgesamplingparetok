
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
  transformed <- .transform_to_normal(samples_4_fit, num_coupling_layers = num_coupling_layers, epochs = epochs, learning_rate = learning_rate, verbose = verbose, seed = seed, return_model = TRUE)

  trained_realnvp <- transformed$model
  if(verbose){
    print(trained_realnvp)
  }
  realnvp_generated <- vector("list", repetitions)
  for (i in seq_len(repetitions)) {
    realnvp_generated[[i]] <- matrix(rnorm(n_post * ncol(samples_4_fit)), nrow = n_post)
    if (verbose) {
        print(paste("Dimension of generated samples in repetition", i, ":", nrow(realnvp_generated[[i]]), ",", ncol(realnvp_generated[[i]])))
    }
  }
  # Calculate q12: log density of the posterior samples under the proposal
  q12 <- apply(samples_4_iter, 1, function(x) {
    x_tensor <- torch_tensor(matrix(x, nrow = 1), dtype = torch_float32())
    # Forward pass through the RealNVP model
    result <- trained_realnvp$forward(x_tensor)
    transformed_sample <- result[[1]]  # This is now a tensor in the normal space
    log_jacobian <- result[[2]]  # Log-Jacobian determinant from the transformation

    # Evaluate the density under the standard multivariate normal
    transformed_sample_array <- as.array(transformed_sample)
    log_density_normal <- dmvnorm(transformed_sample_array, mean = rep(0, length(transformed_sample_array)), sigma = diag(length(transformed_sample_array)), log = TRUE)

    # Combine the normal log-density with the log-Jacobian
    total_log_density <- log_density_normal + as.numeric(log_jacobian)
    total_log_density
  })
  # Calculate q22: log density of the generated samples under the proposal
  q22 <- vector("list", repetitions)
  for (i in seq_len(repetitions)) {
      # Calculate the density of the generated samples directly
      q22[[i]] <- apply(realnvp_generated[[i]], 1, function(x) {
          # Evaluate the density under the standard multivariate normal
          log_density_normal <- dmvnorm(x, mean = rep(0, length(x)), sigma = diag(length(x)), log = TRUE)
          log_density_normal
      })
  }
  # Evaluate q11: log posterior + Jacobian for the posterior samples
  q11 <- apply(samples_4_iter, 1, function(x) {
    posterior_val <- log_posterior(x, data = data, keep_log_eval = keep_log_eval, ...)
    x_tensor <- torch_tensor(matrix(x, nrow = 1), dtype = torch_float32())
    posterior_val
  })
  # Evaluate q21: log posterior + Jacobian for the generated samples
  q21 <- vector("list", repetitions)
  for (i in seq_len(repetitions)) {
    q21[[i]] <- apply(realnvp_generated[[i]], 1, function(x) {
      posterior_val <- log_posterior(x, data = data, keep_log_eval = keep_log_eval, ...)
      x_tensor <- torch_tensor(matrix(x, nrow = 1), dtype = torch_float32())
      jacobian_val <- trained_realnvp$inverse(x_tensor)[[2]]
      jacobian_val <- as.numeric(jacobian_val)
      posterior_val + jacobian_val
    })
  }
  if(verbose) {
    print("summary(q12): (log_dens of proposal (i.e., with dmvnorm) for posterior samples)")
    print(summary(q12))
    print("summary(q22): (log_dens of proposal (i.e., with dmvnorm) for generated samples)")
    print(lapply(q22, summary))
    print("summary(q11): (log_dens of posterior (i.e., with log_posterior) for posterior samples)")
    print(summary(q11))
    print("summary(q21): (log_dens of posterior (i.e., with log_posterior) for generated samples)")
    print(lapply(q21, summary))
    .PROPOSALS <- vector("list", repetitions)
    # for (i in seq_len(repetitions)) {
    #   .PROPOSALS[[i]] <- .invTransform2Real(gen_samples[[i]], lb, ub, param_types)
    # }
    # assign(".PROPOSALS", .PROPOSALS, pos = .GlobalEnv)
    # message("All proposal samples written to .GlobalEnv as .PROPOSALS")
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
      L = NULL,
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
        L = NULL,
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
