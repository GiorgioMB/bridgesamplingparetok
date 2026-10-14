# rstan (through RcppParallel) and the cmdstanr model methods each load a
# shared library with the soname libtbb.so.2, from different builds. When one
# of them is loaded, loading the other in the same R process segfaults in
# dyn.load(), in either order (seen on Linux with rstan 2.36.0.9000,
# RcppParallel 5.1.11.2 and CmdStan 2.40.0; see also stan-dev/cmdstanr#1177).
# These helpers skip a test that would load the second library. They need
# /proc/self/maps, so they never skip on systems without it.

.mapped_tbb <- function() {
  if (!file.exists("/proc/self/maps")) {
    return(character(0))
  }
  maps <- readLines("/proc/self/maps", warn = FALSE)
  paths <- unique(sub("^.* ", "", maps[grepl("/libtbb[^/]*$", maps)]))
  normalizePath(paths, mustWork = FALSE)
}

.rcppparallel_tbb_dir <- function() {
  d <- system.file("lib", package = "RcppParallel")
  if (!nzchar(d)) {
    return(NA_character_)
  }
  normalizePath(d, mustWork = FALSE)
}

.is_rcppparallel_tbb <- function(paths) {
  d <- .rcppparallel_tbb_dir()
  if (is.na(d)) {
    return(rep(FALSE, length(paths)))
  }
  startsWith(paths, paste0(d, "/"))
}

# Use before a test loads rstan, also before skip_if_not_installed("rstan"),
# which loads rstan with requireNamespace().
skip_if_other_tbb_loaded <- function() {
  tbb <- .mapped_tbb()
  if (any(!.is_rcppparallel_tbb(tbb))) {
    testthat::skip(paste(
      "a libtbb that is not from RcppParallel is loaded (cmdstanr model methods);",
      "loading rstan in this R process would segfault"
    ))
  }
}

# Use before a test calls cmdstanr model methods ($init_model_methods(),
# $log_prob(), $grad_log_prob(), $unconstrain_draws()).
skip_if_rcppparallel_tbb_loaded <- function() {
  tbb <- .mapped_tbb()
  if (any(.is_rcppparallel_tbb(tbb))) {
    testthat::skip(paste(
      "RcppParallel's libtbb is loaded (rstan);",
      "cmdstanr model methods in this R process would segfault"
    ))
  }
}
