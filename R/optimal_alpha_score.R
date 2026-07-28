## optimal_alpha_score.R --------------------------------------------------
##
## Closed-form Ledoit-Wolf optimal weight for the hybrid covariance
##
##     Sigma_hybrid = (1 - alpha) * Sigma_n + alpha * Sigma_score,
##
## where Sigma_n is the sample covariance of the (centred) draws and
## Sigma_score = (n^{-1} sum_i (s_i - bar s)(s_i - bar s)^T)^{-1} is the
## score-matched covariance built from per-draw scores
## s_i = grad log p_tilde(theta_i).  Following Ledoit & Wolf (2004,
## J. Multivariate Analysis 88) the alpha that minimises
## E ||Sigma_hybrid - Sigma||_F^2 has the closed form
##
##     alpha* = (V_n - C_{n,score}) / gamma_frob
##
## with
##
##     V_n            = E ||Sigma_n - Sigma||_F^2
##     C_{n,score}    = E < Sigma_n - Sigma, Sigma_score - Sigma >_F
##     gamma_frob     = E ||Sigma_n - Sigma_score||_F^2.
##
## Plug-in estimators use the per-draw rank-1 contributions
##
##     w_i = (theta_i - bar theta)(theta_i - bar theta)^T,
##     T_i = -Sigma_score (s_i - bar s)(s_i - bar s)^T Sigma_score / n
##
## (the second is the leave-one-out / delta-method linearisation of
## Sigma_score = Lambda^{-1} through the matrix inverse).  The
## scalar-target Ledoit-Wolf shrinkage toward (tr Sigma_n / p) * I is also
## returned as a calibration baseline because it requires no gradients.

.optimal_alpha_score <- function(samples_4_fit,
                                 gradients_4_fit,
                                 Sigma_n     = NULL,
                                 Sigma_score = NULL,
                                 include_scalar_baseline = TRUE) {

  if (is.null(samples_4_fit) || is.null(gradients_4_fit))
    stop(".optimal_alpha_score(): need both samples_4_fit and gradients_4_fit.",
         call. = FALSE)
  if (!identical(dim(samples_4_fit), dim(gradients_4_fit)))
    stop(".optimal_alpha_score(): samples_4_fit and gradients_4_fit ",
         "must have the same dimensions.", call. = FALSE)

  ## Drop rows with any non-finite parameter or score value.
  finite_rows <- stats::complete.cases(samples_4_fit) &
                 stats::complete.cases(gradients_4_fit) &
                 apply(is.finite(samples_4_fit),  1, all) &
                 apply(is.finite(gradients_4_fit), 1, all)
  Theta <- samples_4_fit[finite_rows, , drop = FALSE]
  G     <- gradients_4_fit[finite_rows, , drop = FALSE]
  n <- nrow(Theta)
  p <- ncol(Theta)
  if (n <= p + 1L)
    stop(sprintf(
      ".optimal_alpha_score(): need n > p + 1 finite rows (have n = %d, p = %d).",
      n, p), call. = FALSE)

  ## Centred draws and centred scores.
  Theta_c <- sweep(Theta, 2L, colMeans(Theta), check.margin = FALSE)
  S_c     <- sweep(G,     2L, colMeans(G),     check.margin = FALSE)

  if (is.null(Sigma_n))
    Sigma_n <- crossprod(Theta_c) / n
  if (is.null(Sigma_score)) {
    Lambda <- crossprod(S_c) / n
    Lambda <- as.matrix(Matrix::nearPD(Lambda)$mat)
    Sigma_score <- tryCatch(chol2inv(chol(Lambda)),
                            error = function(e) solve(Lambda))
  }

  ## V_n = n^{-2} sum_i ||w_i - Sigma_n||_F^2 with w_i = theta_c_i theta_c_i^T.
  ##
  ## Vectorised:
  ##   sum_i ||w_i||_F^2     = sum_i (theta_c_i' theta_c_i)^2
  ##   sum_i <w_i, Sigma_n>  = sum_i theta_c_i' Sigma_n theta_c_i
  ##   sum_i ||Sigma_n||_F^2 = n * ||Sigma_n||_F^2
  norms2_theta <- rowSums(Theta_c * Theta_c)             # ||theta_c_i||^2
  quad_Sn      <- rowSums((Theta_c %*% Sigma_n) * Theta_c)
  Sn_F2        <- sum(Sigma_n * Sigma_n)
  V_n <- (sum(norms2_theta^2) - 2 * sum(quad_Sn) + n * Sn_F2) / (n^2)

  ## gamma_frob = ||Sigma_n - Sigma_score||_F^2.
  diff_F2    <- sum((Sigma_n - Sigma_score)^2)
  gamma_frob <- diff_F2

  ## C_{n,score} = n^{-2} sum_i <w_i - Sigma_n, T_i - Sigma_score>_F
  ## with T_i = -Sigma_score (s_c_i s_c_i^T) Sigma_score / n
  ## (linearisation of Sigma_score = Lambda^{-1} around delta_Lambda_i =
  ##  (s_c_i s_c_i^T - Lambda) / n; the Lambda part contributes a constant
  ##  in i and is subtracted when we form T_i - Sigma_score).
  ##
  ## <w_i, T_i> = -theta_c_i' Sigma_score (s_c_i s_c_i^T) Sigma_score theta_c_i / n
  ##            = -(s_c_i' Sigma_score theta_c_i)^2 / n.
  ## <w_i, Sigma_score>     = theta_c_i' Sigma_score theta_c_i.
  ## <Sigma_n, T_i>         = -tr(Sigma_n Sigma_score s_c_i s_c_i^T Sigma_score) / n
  ##                        = -(s_c_i' Sigma_score Sigma_n Sigma_score s_c_i) / n.
  ## <Sigma_n, Sigma_score> = sum(Sigma_n * Sigma_score).
  Sscore_theta <- Theta_c %*% Sigma_score                  # n x p
  cross_quad   <- rowSums(S_c * Sscore_theta)              # s_c_i' Sigma_score theta_c_i
  quad_Sscore  <- rowSums((Theta_c %*% Sigma_score) * Theta_c)
  S_n_S        <- Sigma_score %*% Sigma_n %*% Sigma_score
  quad_SnS     <- rowSums((S_c %*% S_n_S) * S_c)
  Sn_Sscore_F  <- sum(Sigma_n * Sigma_score)

  sum_w_T            <- -sum(cross_quad^2) / n            # sum_i <w_i, T_i>
  sum_w_Sscore       <-  sum(quad_Sscore)
  sum_Sn_T           <- -sum(quad_SnS) / n                # sum_i <Sigma_n, T_i>
  sum_Sn_Sscore_term <-  n * Sn_Sscore_F

  C_n_score <- (sum_w_T - sum_w_Sscore - sum_Sn_T + sum_Sn_Sscore_term) / (n^2)

  ## alpha* with degenerate-target guard.
  if (!is.finite(gamma_frob) || gamma_frob <= .Machine$double.eps * Sn_F2) {
    warning(".optimal_alpha_score(): Sigma_n and Sigma_score are essentially ",
            "identical; setting alpha_star = 0.", call. = FALSE)
    alpha_star <- 0
  } else {
    alpha_star <- (V_n - C_n_score) / gamma_frob
    alpha_star <- max(0, min(1, alpha_star))
  }

  ## Scalar-target Ledoit-Wolf baseline (no gradients).
  alpha_star_scalar <- NA_real_
  if (isTRUE(include_scalar_baseline)) {
    mu  <- sum(diag(Sigma_n)) / p
    Sn_minus_muI_F2 <- sum((Sigma_n - mu * diag(p))^2)
    if (is.finite(Sn_minus_muI_F2) &&
        Sn_minus_muI_F2 > .Machine$double.eps * Sn_F2) {
      alpha_star_scalar <- max(0, min(1, V_n / Sn_minus_muI_F2))
    } else {
      alpha_star_scalar <- 0
    }
  }

  list(alpha_star        = alpha_star,
       alpha_star_scalar = alpha_star_scalar,
       V_n               = V_n,
       C_n_score         = C_n_score,
       gamma_frob        = gamma_frob,
       Sigma_n           = Sigma_n,
       Sigma_score       = Sigma_score,
       n_used            = n,
       p                 = p)
}
