
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

.stan_log_posterior <- function(s.row, data, keep_log_eval) {
  out <- tryCatch(rstan::log_prob(object = data$stanfit, upars = s.row), error = function(e) -Inf)
  if (is.na(out)) out <- -Inf
  result <- data.frame(matrix(s.row, nrow = 1))
  result$log_posterior <- out
  csv_file <- "rstan_log_eval.csv"
  # Append the result to the CSV file
  if (keep_log_eval) {
    if (!file.exists(csv_file)) {
      write.csv(result, file = csv_file, row.names = FALSE)
    } else {
      write.table(result, file = csv_file, row.names = FALSE, col.names = FALSE, append = TRUE, sep = ",")
    }
  }
  return(out)
}
                  
.cmdstan_log_posterior <- function(s.row, data, keep_log_eval) {
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
  if (keep_log_eval) {
    if (!file.exists(csv_file)) {
      write.csv(result, file = csv_file, row.names = FALSE)
    } else {
      write.table(result, file = csv_file, row.names = FALSE, col.names = FALSE, append = TRUE, sep = ",")
    }
  }
  
  return(out)
}


                  
#--------------------------------------------------------------------------
# functions for RealNVP support / DEPRECATED
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
    },
    inverse = function(y) {
      print("Divide following by two")
      print(ncol(y))
      print("Expected Value")
      print(d)
      y1 <- y[, 1:d]
      y2 <- y[, (d + 1):input_shape]
      
      st <- self$scale_translation_network(y1)
      s <- st[, 1:d]
      t <- st[, (d + 1):(2 * d)]
      
      x1 <- y1
      x2 <- (y2 - t) / torch_exp(s)
      
      log_det_jacobian <- -torch_sum(s, dim = 2)
      
      output <- torch_cat(list(x1, x2), dim = 2)
      list(output, log_det_jacobian)
    }
  )
  
  net()
}


.create_realnvp <- function(input_shape, num_coupling_layers) {
  net <- nn_module(
    initialize = function() {
      self$coupling_layers <- nn_module_list()
      for (i in seq_len(num_coupling_layers)) {
        self$coupling_layers$append(.create_affine_coupling_layer(input_shape))
      }
    },
    forward = function(x) {
      total_log_det_jacobian <- 0
      for (i in seq_len(length(self$coupling_layers))) {
        result <- self$coupling_layers[[i]](x)
        x <- result[[1]]
        total_log_det_jacobian <- total_log_det_jacobian + result[[2]]
      }
      list(x, total_log_det_jacobian)
    },
    inverse = function(y) {
      total_log_det_jacobian <- 0
      for (i in length(self$coupling_layers):1) {
        result <- self$coupling_layers[[i]]$inverse(y)
        y <- result[[1]]
        total_log_det_jacobian <- total_log_det_jacobian + result[[2]]
      }
      list(y, total_log_det_jacobian)
    }
  )
  
  net()
}


.negative_log_likelihood <- function(y_true, y_pred) {
  z <- y_pred[[1]]
  log_det_jacobian <- y_pred[[2]]
  logp_z <- -0.5 * torch_sum(z^2, dim = 2L) - 0.5 * ncol(z) * log(2 * pi)
  nll <- -torch_mean(logp_z + log_det_jacobian)
}

.train_realnvp <- function(samples, normal_samples, num_coupling_layers = 5, epochs = 50, learning_rate = 0.001, verbose = FALSE) {
  library(torch)
  input_shape <- ncol(samples)
  realnvp_model <- .create_realnvp(input_shape, num_coupling_layers)
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
                 
.transform_to_normal <- function(samples, num_coupling_layers = 5, epochs = 50, learning_rate = 0.001, verbose = FALSE, seed = 1, return_model = FALSE) {
    set.seed(seed)
    n <- nrow(samples)
    d <- ncol(samples)

    # Apply column-wise normalization
    column_means <- colMeans(samples)
    column_sds <- apply(samples, 2, sd)
    samples <- sweep(samples, 2, column_means, "-")
    samples <- sweep(samples, 2, column_sds, "/")
    if (!inherits(samples, "torch_tensor")) {
        samples <- torch::torch_tensor(samples)
    }

    normal_samples <- rnorm(n * d)
    normal_samples <- matrix(normal_samples, nrow = n, ncol = d)
    normal_samples <- torch::torch_tensor(normal_samples)

    trained_rnvp <- .train_realnvp(samples, normal_samples, num_coupling_layers, epochs, learning_rate, verbose)
    x_rnvp_warped <- as.matrix(trained_rnvp$forward(samples)[[1]])
    x_log_det_jacobian <- as.matrix(trained_rnvp$forward(samples)[[2]])

    if (return_model) {
        return(list(transformed_samples = x_rnvp_warped, log_det_jacobian = x_log_det_jacobian, model = trained_rnvp))
    } else {
        return(list(x_rnvp_warped, x_log_det_jacobian))
    }

}
