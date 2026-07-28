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
  keep_log_eval = FALSE,
  ## Score-matching proposal-fit (gradient-based) options.
  ## gradients_4_fit: optional n_fit x p matrix of score vectors
  ##                  s(theta) = grad log p(theta) at samples_4_fit rows.
  ## proposal_fit: "sample" (default, current behaviour), "score"
  ##               (Sigma^-1 = E[s s^T]), "hybrid" (matrix geometric
  ##               mean of V_sample and V_score; the Fisher-divergence-
  ##               optimal dense combiner of Seyboldt-Carlson-Carpenter
  ##               2026; no weight parameter), or "hybrid_arith"
  ##               (arithmetic-mean convex combo of V_sample and
  ##               V_score, weighted by alpha_score; the original
  ##               hybrid combiner from the dev fork, retained as an
  ##               ablation baseline).
  ## alpha_score: convex-combo weight on score estimator for
  ##              proposal_fit = "hybrid_arith". Either a numeric in
  ##              [0,1] (fixed weight; default 0.5), or NULL meaning
  ##              "compute the closed-form Ledoit-Wolf optimal weight
  ##              per fit via .optimal_alpha_score()". Ignored for
  ##              proposal_fit in ("sample","score","hybrid").
  gradients_4_fit = NULL,
  proposal_fit    = c("sample", "score", "hybrid", "hybrid_arith"),
  alpha_score     = 0.5) {

  proposal_fit <- match.arg(proposal_fit)
  # keep the original name (if given as one) for clusterExport below
  log_posterior_name <- if (is.character(log_posterior)) log_posterior else NULL
  log_posterior <- .wrap_log_posterior(log_posterior, envir = envir)
  alpha_score_input <- alpha_score
  if (!is.null(alpha_score) &&
      !(is.numeric(alpha_score) && length(alpha_score) == 1L &&
        is.finite(alpha_score) && alpha_score >= 0 && alpha_score <= 1))
    stop("alpha_score must be NULL or a single numeric in [0, 1].",
         call. = FALSE)
  if (is.null(neff))
    neff <- nrow(samples_4_iter)

  n_post <- nrow(samples_4_iter)

  # get mean & covariance matrix and generate samples from proposal
  m <- apply(samples_4_fit, 2, mean)
  V_sample <- cov(samples_4_fit)

  use_gradients <- !is.null(gradients_4_fit) && proposal_fit != "sample"
  if (use_gradients) {
    if (nrow(gradients_4_fit) != nrow(samples_4_fit) ||
        ncol(gradients_4_fit) != ncol(samples_4_fit)) {
      warning("gradients_4_fit dims do not match samples_4_fit; ",
              "falling back to sample covariance.", call. = FALSE)
      use_gradients <- FALSE
    }
  }

  ## Initialise proposal_fit_info; populated below if gradients are used.
  ## combiner records how V_sample and V_score were combined:
  ##   "none"       : proposal_fit = "sample" (V = V_sample)
  ##   "score_only" : proposal_fit = "score"  (V = V_score)
  ##   "arithmetic" : proposal_fit = "hybrid" (V = (1-a) V_n + a V_score)
  ##   "geometric"  : proposal_fit = "hybrid_geom" (V = V_n # V_score)
  ##   "geometric_fallback_arithmetic" : "hybrid_geom" path took the
  ##                  PSD fallback inside .geometric_mean_psd().
  proposal_fit_info <- list(
    proposal_fit       = proposal_fit,
    combiner           = NA_character_,
    alpha_score_input  = alpha_score_input,
    alpha_score_used   = NA_real_,
    alpha_star         = NA_real_,
    alpha_star_scalar  = NA_real_,
    V_n                = NA_real_,
    C_n_score          = NA_real_,
    gamma_frob         = NA_real_,
    scores_centered    = NA
  )

  if (use_gradients) {
    ## Score-matching estimate of inverse covariance using *centred*
    ## scores (s_i - bar s)(s_i - bar s)^T. The uncentred form retains an
    ## O(bar s bar s^T) finite-sample bias even though E_pi[s] = 0.
    finite_rows <- stats::complete.cases(gradients_4_fit) &
                   apply(is.finite(gradients_4_fit), 1, all)
    if (sum(finite_rows) < ncol(gradients_4_fit) + 1L) {
      warning(sprintf(
        "only %d of %d gradient rows finite (need > ncol = %d); ",
        sum(finite_rows), nrow(gradients_4_fit), ncol(gradients_4_fit)),
        "falling back to sample covariance.", call. = FALSE)
      use_gradients <- FALSE
    } else {
      G_raw      <- gradients_4_fit[finite_rows, , drop = FALSE]
      G          <- sweep(G_raw, 2L, colMeans(G_raw), check.margin = FALSE)
      Prec_score <- crossprod(G) / nrow(G)
      Prec_score <- as.matrix(nearPD(Prec_score)$mat)
      V_score    <- tryCatch(chol2inv(chol(Prec_score)),
                             error = function(e) solve(Prec_score))
      proposal_fit_info$scores_centered <- TRUE
    }
  }
  if (use_gradients) {
    if (proposal_fit == "score") {
      V_tmp <- V_score
      proposal_fit_info$alpha_score_used <- 1
      proposal_fit_info$combiner         <- "score_only"
    } else if (proposal_fit == "hybrid") {
      ## Matrix geometric mean of V_sample and V_score (Seyboldt-
      ## Carlson-Carpenter 2026, dense Fisher-divergence-optimal
      ## combiner). Default combiner since 2026-05. No alpha; the
      ## geometric mean is uniquely determined by the two PSD inputs.
      ## We pre-nearPD V_sample here so .geometric_mean_psd() sees
      ## two PSD matrices; the Prec_score branch above already
      ## nearPD'd V_score.
      V_sample_pd <- as.matrix(nearPD(V_sample)$mat)
      V_tmp <- .geometric_mean_psd(V_sample_pd, V_score)
      proposal_fit_info$alpha_score_used <- NA_real_
      proposal_fit_info$combiner <-
        if (isTRUE(attr(V_tmp, "fallback"))) "geometric_fallback_arithmetic"
        else                                  "geometric"
      attr(V_tmp, "fallback") <- NULL
      if (verbose) {
        cat(sprintf("[proposal_fit=hybrid] combiner = %s\n",
                    proposal_fit_info$combiner))
      }
    } else { # "hybrid_arith"
      if (is.null(alpha_score)) {
        ## Closed-form Ledoit-Wolf optimal alpha. The helper recomputes
        ## Sigma_n and Sigma_score on the same finite-row subset so that
        ## V_n / C_n_score / gamma_frob are exactly consistent with the
        ## V_sample and V_score used here.
        Theta_fit <- samples_4_fit[finite_rows, , drop = FALSE]
        opt <- .optimal_alpha_score(samples_4_fit   = Theta_fit,
                                    gradients_4_fit = G_raw,
                                    Sigma_n         = V_sample,
                                    Sigma_score     = V_score,
                                    include_scalar_baseline = TRUE)
        a <- opt$alpha_star
        proposal_fit_info$alpha_star        <- opt$alpha_star
        proposal_fit_info$alpha_star_scalar <- opt$alpha_star_scalar
        proposal_fit_info$V_n               <- opt$V_n
        proposal_fit_info$C_n_score         <- opt$C_n_score
        proposal_fit_info$gamma_frob        <- opt$gamma_frob
        if (verbose)
          cat(sprintf("[proposal_fit=hybrid_arith] alpha_star = %.4f (scalar %.4f)\n",
                      opt$alpha_star, opt$alpha_star_scalar))
      } else {
        a <- max(0, min(1, alpha_score))
        ## Even with a fixed weight, compute the optimal-alpha breakdown
        ## as a free diagnostic so callers can compare a vs alpha_star.
        Theta_fit <- samples_4_fit[finite_rows, , drop = FALSE]
        opt <- tryCatch(
          .optimal_alpha_score(samples_4_fit   = Theta_fit,
                               gradients_4_fit = G_raw,
                               Sigma_n         = V_sample,
                               Sigma_score     = V_score,
                               include_scalar_baseline = TRUE),
          error = function(e) NULL,
          warning = function(w) NULL)
        if (!is.null(opt)) {
          proposal_fit_info$alpha_star        <- opt$alpha_star
          proposal_fit_info$alpha_star_scalar <- opt$alpha_star_scalar
          proposal_fit_info$V_n               <- opt$V_n
          proposal_fit_info$C_n_score         <- opt$C_n_score
          proposal_fit_info$gamma_frob        <- opt$gamma_frob
        }
      }
      V_tmp <- a * V_score + (1 - a) * V_sample
      proposal_fit_info$alpha_score_used <- a
      proposal_fit_info$combiner         <- "arithmetic"
    }
    if (verbose) {
      cat(sprintf("[proposal_fit=%s] eigenvalue ranges\n", proposal_fit))
      cat(sprintf("  sample Sigma : %.3g .. %.3g\n",
                  min(eigen(V_sample, symmetric = TRUE,
                            only.values = TRUE)$values),
                  max(eigen(V_sample, symmetric = TRUE,
                            only.values = TRUE)$values)))
      cat(sprintf("  score Sigma  : %.3g .. %.3g\n",
                  min(eigen(V_score, symmetric = TRUE,
                            only.values = TRUE)$values),
                  max(eigen(V_score, symmetric = TRUE,
                            only.values = TRUE)$values)))
    }
  } else {
    V_tmp <- V_sample
    proposal_fit_info$alpha_score_used <- 0
    proposal_fit_info$combiner         <- "none"
  }
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
      } else if (!is.null(log_posterior_name)) {
        parallel::clusterExport(cl = cl, varlist = log_posterior_name, envir = envir)
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
              pareto_k_deni = pareto_k_deni, pareto_k_inv_deni = pareto_k_inv_deni, mcse_logml = std_logmls,
              proposal_fit_info = proposal_fit_info)
    class(out) <- "bridge"
  } else if (repetitions > 1) {
    out <- list(logml = logml, niter = niter, method = "normal", repetitions = repetitions, numi = numi, deni = deni,
              pareto_k_numi = pareto_k_numi, pareto_k_deni = pareto_k_deni, pareto_k_inv_deni = pareto_k_inv_deni, 
              mcse_logml = std_logmls, proposal_fit_info = proposal_fit_info)
    class(out) <- "bridge_list"
  }

  return(out)

}
