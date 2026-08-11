## Matrix geometric mean of two positive (semi-)definite covariance
## matrices: A # B = A^{1/2} (A^{-1/2} B A^{-1/2})^{1/2} A^{1/2}.
##
## Reference: Seyboldt, Carlson, Carpenter (2026), "Preconditioning
## Hamiltonian Monte Carlo by minimizing Fisher Divergence". The
## dense Fisher-divergence-optimal preconditioner is the matrix
## geometric mean of the sample covariance of the draws and the
## inverse covariance of the scores; the helper here is the
## bridge-sampling analogue used for `proposal_fit = "hybrid_geom"`.
##
## The geometric mean is the (affine-invariant Fisher-Rao)
## Riemannian midpoint of A and B on the positive-definite cone.
## It coincides with the arithmetic mean (1-a) A + a B at a = 0.5
## when A and B commute (e.g. when both equal the true Sigma for a
## Gaussian target), and behaves more gracefully than the arithmetic
## mean when A and B disagree in their orientation or scale.
##
## Implementation: symmetric eigendecomposition of A, then of the
## inner sandwich M = A^{-1/2} B A^{-1/2}, with PSD clipping on tiny
## (round-off) negative eigenvalues. On failure (rank-deficient A,
## non-finite outputs) the function falls back to the arithmetic
## mean 0.5 (A + B) with a single warning and returns it tagged so
## callers can record the fallback path.
##
## Arguments:
##   A, B: p x p symmetric matrices, assumed positive semi-definite.
##         A and B should already be nearPD-corrected by the caller
##         when they are sample/score covariance estimates.
##   eps:  relative tolerance for clipping tiny negative eigenvalues
##         introduced by round-off (default sqrt(.Machine$double.eps)).
##
## Returns: a p x p symmetric matrix; attribute "fallback" is TRUE
##   when the arithmetic-mean fallback path was taken.
.geometric_mean_psd <- function(A, B, eps = sqrt(.Machine$double.eps)) {
  if (!is.matrix(A) || !is.matrix(B))
    stop(".geometric_mean_psd(): A and B must be matrices.", call. = FALSE)
  if (!identical(dim(A), dim(B)) || nrow(A) != ncol(A))
    stop(".geometric_mean_psd(): A and B must be square and the same size.",
         call. = FALSE)

  ## Force exact symmetry up front -- both eigen() and the
  ## sandwiches below assume it.
  A <- (A + t(A)) / 2
  B <- (B + t(B)) / 2

  arithmetic_fallback <- function(reason) {
    warning(sprintf(".geometric_mean_psd(): %s; falling back to ",
                    reason), "arithmetic mean.", call. = FALSE)
    out <- 0.5 * (A + B)
    out <- (out + t(out)) / 2
    attr(out, "fallback") <- TRUE
    out
  }

  eigA <- tryCatch(eigen(A, symmetric = TRUE),
                   error = function(e) NULL)
  if (is.null(eigA) || any(!is.finite(eigA$values)))
    return(arithmetic_fallback("eigendecomposition of A failed"))

  ## Clip tiny negative eigenvalues from round-off. We do NOT bail
  ## out on small positive eigenvalues: the caller pre-regularises
  ## A via Matrix::nearPD(), so A is PD by construction; small
  ## eigenvalues are real and the geometric mean is mathematically
  ## well-defined. The post-hoc non-finite check below catches any
  ## genuine numerical failure (e.g. exact zeros via the pseudo-
  ## inverse guard on invsL).
  lam <- eigA$values
  lam_max <- max(abs(lam))
  if (lam_max <= 0)
    return(arithmetic_fallback("A is the zero matrix"))
  lam <- pmax(lam, 0)

  U     <- eigA$vectors
  sqrtL <- sqrt(lam)
  invsL <- ifelse(sqrtL > 0, 1 / sqrtL, 0)    # pseudo-inverse for true zeros
  A_half     <- U %*% (sqrtL * t(U))
  A_neg_half <- U %*% (invsL * t(U))

  M <- A_neg_half %*% B %*% A_neg_half
  M <- (M + t(M)) / 2

  eigM <- tryCatch(eigen(M, symmetric = TRUE),
                   error = function(e) NULL)
  if (is.null(eigM) || any(!is.finite(eigM$values)))
    return(arithmetic_fallback("eigendecomposition of A^{-1/2} B A^{-1/2} failed"))

  ## Same policy on the inner sandwich M: clip round-off negatives,
  ## otherwise trust the eigendecomposition. The post-hoc non-finite
  ## check below catches genuine numerical failures.
  mu <- eigM$values
  mu_max <- max(abs(mu))
  if (mu_max <= 0)
    return(arithmetic_fallback("A^{-1/2} B A^{-1/2} is the zero matrix"))
  mu <- pmax(mu, 0)

  V    <- eigM$vectors
  M_half <- V %*% (sqrt(mu) * t(V))

  G <- A_half %*% M_half %*% A_half
  G <- (G + t(G)) / 2

  if (any(!is.finite(G)))
    return(arithmetic_fallback("non-finite entries in A # B"))

  attr(G, "fallback") <- FALSE
  G
}
