
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
.create_affine_coupling_layer <- function(input_shape) {
  d <- as.integer(input_shape / 2)

  scale_translation_network <- nn_sequential(
    nn_linear(d, 512),
    nn_relu(),
    nn_linear(512, d * 2)
  )
  
  net <- nn_module(
    initialize = function() {
      self$scale_translation_network <- scale_translation_network
    },
    forward = function(x) {
      x1 <- x[, 1:d]
      x2 <- x[, (d + 1):input_shape]
      
      st <- self$scale_translation_network(x1)
      s <- st[, 1:d]
      t <- st[, (d + 1):(2 * d)]
      
      y1 <- x1
      y2 <- x2 * torch_exp(s) + t
      
      log_det_jacobian <- torch_sum(s, dim = 2) 
      
      output <- torch_cat(list(y1, y2), dim = 2)
      list(output, log_det_jacobian)
    }
  )
  
  net
}

.create_realnvp <- function(input_shape, num_coupling_layers) {
  net <- nn_module(
    initialize = function() {
      self$coupling_layers <- nn_module_list(lapply(seq_len(num_coupling_layers), function(i) .create_affine_coupling_layer(input_shape)))
      self$register_parameters()  # Register parameters
    },
    forward = function(x) {
      total_log_det_jacobian <- 0
      
      for (coupling_layer in self$coupling_layers) {
        result <- coupling_layer(x)
        x <- result[[1]]  # updated x
        total_log_det_jacobian <- total_log_det_jacobian + result[[2]]
      }
      
      list(x, total_log_det_jacobian)
    }
  )
  
  net
}

.negative_log_likelihood <- function(y_true, y_pred) {
  z <- y_pred[[1]]
  log_det_jacobian <- y_pred[[2]]
  logp_z <- -0.5 * torch_sum(z^2, dim = 2L) - 0.5 * ncol(z) * log(2 * pi)
  nll <- -torch_mean(logp_z + log_det_jacobian)
  
  nll
}

# Define the training function for RealNVP
.train_realnvp <- function(samples, normal_samples, num_coupling_layers = 5, epochs = 50, batch_size = 32, learning_rate = 0.001, train_ratio = 0.8, verbose = FALSE) {
  input_shape <- ncol(samples)
  realnvp_model <- .create_realnvp(input_shape, num_coupling_layers)
  print(cat("Parameters are", realnvp_model$parameters)
  optimizer <- optim_adam(realnvp_model$parameters, lr = learning_rate)
  
  for (epoch in seq_len(epochs)) {
    realnvp_model$zero_grad()
    outputs <- realnvp_model(samples)
    loss <- .negative_log_likelihood(normal_samples, outputs)
    loss$backward()
    optimizer$step()
    
    if(verbose){
      if (epoch %% 10 == 0) {
        cat(sprintf("Epoch [%d/%d], Loss: %f\n", epoch, epochs, loss$item()))
      }
    }
  }
  
  realnvp_model
}
