
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

.stan_log_posterior <- function(s.row, data) {
  out <- tryCatch(rstan::log_prob(object = data$stanfit, upars = s.row), error = function(e) -Inf)
  if (is.na(out)) out <- -Inf
  result <- data.frame(matrix(s.row, nrow = 1))
  result$log_posterior <- out
  csv_file <- "rstan_log_eval.csv"
  # Append the result to the CSV file
  if (!file.exists(csv_file)) {
    write.csv(result, file = csv_file, row.names = FALSE)
  } else {
    write.table(result, file = csv_file, row.names = FALSE, col.names = FALSE, append = TRUE, sep = ",")
  }
  return(out)
}
                  
.cmdstan_log_posterior <- function(s.row, data) {
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
  if (!file.exists(csv_file)) {
    write.csv(result, file = csv_file, row.names = FALSE)
  } else {
    write.table(result, file = csv_file, row.names = FALSE, col.names = FALSE, append = TRUE, sep = ",")
  }
  
  return(out)
}

#--------------------------------------------------------------------------
# functions for RealNVP support
#--------------------------------------------------------------------------
,create_affine_coupling_layer <- function(input_shape) {
  # Define the input layer
  input <- nnf_input(input_shape)
  
  # Split the input tensor
  d <- as.integer(input_shape / 2)
  x1 <- nn_lambda(function(x) x[, 1:d], output_shape = c(d))
  x2 <- nn_lambda(function(x) x[, (d + 1):input_shape], output_shape = c(d))
  
  # Scale and translation network
  scale_translation_network <- nn_sequential(
    nn_linear(d, 512),
    nn_relu(),
    nn_linear(512, d * 2)
  )
  
  st <- scale_translation_network(x1)
  s <- nn_lambda(function(x) x[, 1:d], output_shape = c(d))(st)
  t <- nn_lambda(function(x) x[, (d + 1):(2 * d)], output_shape = c(d))(st)
  
  y1 <- x1
  y2 <- x2 * exp(s) + t
  
  # Compute the log determinant of the Jacobian
  log_det_jacobian <- torch_sum(s, dim = 2)
  
  # Concatenate y1 and y2
  output <- nn_concatenate(dim = 2)(list(y1, y2))
  
  model <- nn_module(
    initialize = function() {
      self$input <- input
      self$output <- output
    },
    forward = function(x) {
      list(self$output(x), log_det_jacobian)
    }
  )
  
  return(model)
}

.create_realnvp <- function(input_shape, num_coupling_layers) {
  inputs <- nnf_input(input_shape)
  x <- inputs
  total_log_det_jacobian <- 0
  
  for (i in seq_len(num_coupling_layers)) {
    coupling_layer <- .create_affine_coupling_layer(input_shape)
    result <- coupling_layer(x)
    x <- result[[1]]  # updated x
    total_log_det_jacobian <- total_log_det_jacobian + result[[2]]  # accumulate the log determinant of Jacobian
  }
  
  model <- nn_module(
    initialize = function() {
      self$inputs <- inputs
      self$x <- x
      self$total_log_det_jacobian <- total_log_det_jacobian
    },
    forward = function(x) {
      list(self$x(x), self$total_log_det_jacobian)
    }
  )
  
  return(model)
}

.negative_log_likelihood <- function(y_true, y_pred) {
  z <- y_pred[[1]]
  log_det_jacobian <- y_pred[[2]]
  logp_z <- -0.5 * torch_sum(z^2, dim = 1L) - 0.5 * length(z) * log(2 * pi)
  nll <- -mean(logp_z + log_det_jacobian)
  return(nll)
}

.train_realnvp <- function(samples, normal_samples, num_coupling_layers = 5, epochs = 50, batch_size = 32, learning_rate = 0.001, train_ratio = 0.8) {
  library(torch)
  input_shape <- ncol(samples)
  realnvp_model <- .create_realnvp(input_shape, num_coupling_layers)
  optimizer <- optim_adam(realnvp_model$parameters, lr = learning_rate)
  
  dataset <- tensor_dataset(list(samples, normal_samples))
  dataloader <- dataloader(dataset, batch_size = batch_size, shuffle = TRUE)
  
  # Training loop
  for (epoch in seq_len(epochs)) {
    for (batch in enumerate(dataloader)) {
      optimizer$zero_grad()
      output <- realnvp_model(batch[[1]])
      loss <- .negative_log_likelihood(batch[[2]], output)
      loss$backward()
      optimizer$step()
    }
  }
  
  return(realnvp_model)
}                                      
                                         
