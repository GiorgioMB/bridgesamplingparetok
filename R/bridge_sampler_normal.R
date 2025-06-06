.bridge.sampler.normal <- function(
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
  pareto_smoothing_all,
  pareto_smoothing_last,
  maxiter,
  silent,
  verbose,
  r0,
  tol1,
  tol2,
  return_always,
  use_ess = FALSE,
  calculate_covariance = FALSE,
  keep_log_eval = FALSE) {
  
  if (is.null(neff))
    neff <- nrow(samples_4_iter)

  n_post <- nrow(samples_4_iter)

  # get mean & covariance matrix and generate samples from proposal
  m <- apply(samples_4_fit, 2, mean)
  V_tmp <- cov(samples_4_fit)
  V <- as.matrix(nearPD(V_tmp)$mat) # make sure that V is positive-definite

  # sample from multivariate normal distribution and evaluate for posterior samples and generated samples
  q12 <- dmvnorm(samples_4_iter, mean = m, sigma = V, log = TRUE)
  gen_samples <- vector(mode = "list", length = repetitions)
  q22 <- vector(mode = "list", length = repetitions)
  for (i in seq_len(repetitions)) {
    gen_samples[[i]] <- rmvnorm(n_post, mean = m, sigma = V)
    colnames(gen_samples[[i]]) <- colnames(samples_4_iter)
    q22[[i]] <- dmvnorm(gen_samples[[i]], mean = m, sigma = V, log = TRUE)
  }

  # evaluate log of likelihood times prior for posterior samples and generated samples
  q21 <- vector(mode = "list", length = repetitions)
  if (cores == 1) {
    q11 <- apply(.invTransform2Real(samples_4_iter, lb, ub, param_types), 1, log_posterior,
                 data = data, keep_log_eval = keep_log_eval, ...) + .logJacobian(samples_4_iter, transTypes, lb, ub)
    for (i in seq_len(repetitions)) {
      q21[[i]] <- apply(.invTransform2Real(gen_samples[[i]], lb, ub, param_types), 1, log_posterior,
                        data = data, keep_log_eval = keep_log_eval, ...) + .logJacobian(gen_samples[[i]], transTypes, lb, ub)
    }
  } else if (cores > 1) {
    if (.Platform$OS.type == "unix") {
      split1 <- .split_matrix(matrix = .invTransform2Real(samples_4_iter, lb, ub, param_types), cores = cores)
      q11 <- parallel::mclapply(split1, FUN =
                                  function(x) apply(x, 1, log_posterior, data = data, keep_log_eval = keep_log_eval, ...),
                                  mc.preschedule = FALSE,
                                  mc.cores = cores)
      q11 <- unlist(q11) + .logJacobian(samples_4_iter, transTypes, lb, ub)
      for (i in seq_len(repetitions)) {
        split2 <- .split_matrix(matrix = .invTransform2Real(gen_samples[[i]], lb, ub, param_types), cores = cores)
        q21[[i]] <- parallel::mclapply(split2, FUN =
                                    function(x) apply(x, 1, log_posterior, data = data, keep_log_eval = keep_log_eval, ...),
                                    mc.preschedule = FALSE,
                                    mc.cores = cores)
        q21[[i]] <- unlist(q21[[i]]) + .logJacobian(gen_samples[[i]], transTypes, lb, ub)
      }
    } else {
      cl <- parallel::makeCluster(cores, useXDR = FALSE)
      sapply(packages, function(x) parallel::clusterCall(cl = cl, "require", package = x,
                                                         character.only = TRUE))
      parallel::clusterExport(cl = cl, varlist = varlist, envir = envir)
  
      if (!is.null(rcppFile)) {
        parallel::clusterExport(cl = cl, varlist = "rcppFile", envir = parent.frame())
        parallel::clusterCall(cl = cl, "require", package = "Rcpp", character.only = TRUE)
        parallel::clusterEvalQ(cl = cl, Rcpp::sourceCpp(file = rcppFile))
      } else if (is.character(log_posterior)) {
        parallel::clusterExport(cl = cl, varlist = log_posterior, envir = envir)
      }
  
      q11 <- parallel::parRapply(cl = cl, x = .invTransform2Real(samples_4_iter, lb, ub, param_types), log_posterior,
                                 data = data, keep_log_eval = keep_log_eval, ...) + .logJacobian(samples_4_iter, transTypes, lb, ub)
      for (i in seq_len(repetitions)) {
        q21[[i]] <- parallel::parRapply(cl = cl, x = .invTransform2Real(gen_samples[[i]], lb, ub, param_types), log_posterior,
                                        data = data, keep_log_eval = keep_log_eval, ...) + .logJacobian(gen_samples[[i]], transTypes, lb, ub)
      }
      parallel::stopCluster(cl)
    }
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
  if (any(is.infinite(q11))) {
    warning(sum(is.infinite(q11)), " of the ", length(q11)," log_prob() evaluations on the posterior draws produced -Inf/Inf.", call. = FALSE)
  }
  for (i in seq_len(repetitions)) {
    if (any(is.infinite(q21[[i]]))) {
      warning(sum(is.infinite(q21[[i]])), " of the ", length(q21[[i]])," log_prob() evaluations on the proposal draws produced -Inf/Inf.", call. = FALSE)
    }
  }
  if (any(is.na(q11))) {
    warning(sum(is.na(q11)), " evaluation(s) of log_prob() on the posterior draws produced NA and have been replaced by -Inf.", call. = FALSE)
    q11[is.na(q11)] <- -Inf
  }
  for (i in seq_len(repetitions)) {
    if (all(is.na(q21[[i]]))) {
      stop("Evaluations of log_prob() on all proposal draws produced NA.\n",
           "E.g., rounded to 3 digits (use verbose = TRUE for all proposal samples):\n",
           deparse(round(
             .invTransform2Real(gen_samples[[i]], lb, ub, param_types)[1,],
             3), width.cutoff = 500L),
           call. = FALSE)
    }
    if (any(is.na(q21[[i]]))) {
      warning(sum(is.na(q21[[i]])), " evaluation(s) of log_prob() on the proposal draws produced NA and have been replaced by -Inf.", call. = FALSE)
      q21[[i]][is.na(q21[[i]])] <- -Inf
    }
  }
  logml <- numeric(repetitions)
  niter <- numeric(repetitions)
  std_logmls <- numeric(repetitions)
  pareto_k_numi <- list()
  pareto_k_deni <- list()
  pareto_k_inv_deni <- list()
  numi <- list()
  deni <- list()
  # run iterative updating scheme to compute log of marginal likelihood
  for (i in seq_len(repetitions)) {
    tmp <- .run.iterative.scheme(q11 = q11, q12 = q12, q21 = q21[[i]], q22 = q22[[i]],
                                 r0 = r0, tol = tol1, L = NULL, method = "normal",pareto_smoothing_all = pareto_smoothing_all, use_ess = use_ess,
                                 maxiter = maxiter, silent = silent, pareto_smoothing_last = pareto_smoothing_last, verbose = verbose,
                                 criterion = "r", neff = neff, return_always = return_always, calculate_covariance = calculate_covariance)
    if (!is.null(tmp$r_vals)) {
      warning("logml could not be estimated within maxiter, rerunning with adjusted starting value. \nEstimate might be more variable than usual.", call. = FALSE)
      lr <- length(tmp$r_vals)
      # use geometric mean as starting value
      r0_2 <- sqrt(tmp$r_vals[[lr - 1]] * tmp$r_vals[[lr]])
      tmp <- .run.iterative.scheme(q11 = q11, q12 = q12, q21 = q21[[i]], q22 = q22[[i]], use_ess = use_ess,
                                   r0 = r0_2, tol = tol2, L = NULL, method = "normal", pareto_smoothing_all = pareto_smoothing_all, verbose = verbose,
                                   maxiter = maxiter, silent = silent, return_always = return_always, pareto_smoothing_last = pareto_smoothing_last,
                                   criterion = "logml", neff = neff, calculate_covariance = calculate_covariance)
      tmp$niter <- maxiter + tmp$niter
    }
    logml[i] <- tmp$logml
    niter[i] <- tmp$niter
    std_logmls[i] <- tmp$std_logml
    numi[[i]] <- tmp$numi
    deni[[i]] <- tmp$deni
    if("pareto_k" %in% names(tmp)) {
      if(verbose){
        print(tmp$pareto_k)
      }
      pareto_k_numi[[i]] <- tmp$pareto_k$numi
      pareto_k_deni[[i]] <- tmp$pareto_k$deni
      pareto_k_inv_deni[[i]] <- tmp$pareto_k$inv_deni
    } else {
      if(verbose){
        print("There was an error computing the pareto_k diagnostic")
      }
      pareto_k_numi[[i]] <- NA
      pareto_k_deni[[i]] <- NA
      pareto_k_inv_deni[[i]] <- NA
    }
      
    if (niter[i] == maxiter)
      warning("logml could not be estimated within maxiter.", call. = FALSE)
  }

  if (repetitions == 1) {
    out <- list(logml = logml, niter = niter, method = "normal", q11 = q11, numi = numi, deni = deni,
              q12 = q12, q21 = q21[[1]], q22 = q22[[1]], pareto_k_numi = pareto_k_numi, 
              pareto_k_deni = pareto_k_deni, pareto_k_inv_deni = pareto_k_inv_deni, mcse_logml = std_logmls)
    class(out) <- "bridge"
  } else if (repetitions > 1) {
    out <- list(logml = logml, niter = niter, method = "normal", repetitions = repetitions, numi = numi, deni = deni,
              pareto_k_numi = pareto_k_numi, pareto_k_deni = pareto_k_deni, pareto_k_inv_deni = pareto_k_inv_deni, 
              mcse_logml = std_logmls)
    class(out) <- "bridge_list"
  }

  return(out)

}
