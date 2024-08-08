
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
  input <- tensorflow::tf$keras$layers$Input(shape = input_shape)
  d <- as.integer(input_shape / 2)
  x1 <- tensorflow::tf$keras$layers$Lambda(function(x) x[, 1:d])(input)
  x2 <- tensorflow::tf$keras$layers$Lambda(function(x) x[, (d + 1):input_shape])(input)
  
  scale_translation_network <- tensorflow::tf$keras$models$Sequential() %>%
    tensorflow::tf$keras$layers$Dense(units = 512, activation = 'relu') %>%
    tensorflow::tf$keras$layers$Dense(units = d * 2)

  st <- scale_translation_network(x1)
  s <- tensorflow::tf$keras$layers$Lambda(function(x) x[, 1:d])(st)
  t <- tensorflow::tf$keras$layers$Lambda(function(x) x[, (d + 1):(2 * d)])(st)

  y1 <- x1
  y2 <- x2 * tensorflow::tf$exp(s) + t
  
  log_det_jacobian <- tensorflow::tf$reduce_sum(s, axis = 1)  # sum log(s) for Jacobian determinant
  
  output <- tensorflow::tf$keras$layers$Concatenate()([y1, y2])
  return(tensorflow::tf$keras$Model(inputs = input, outputs = list(output, log_det_jacobian)))
}
                                          
.create_realnvp <- function(input_shape, num_coupling_layers) {
  inputs <- tensorflow::tf$keras$layers$Input(shape = input_shape)
  x <- inputs
  total_log_det_jacobian <- 0

  for (i in seq_len(num_coupling_layers)) {
    coupling_layer <- .create_affine_coupling_layer(input_shape)
    result <- coupling_layer(x)
    x <- result[[1]]  # updated x
    total_log_det_jacobian <- total_log_det_jacobian + result[[2]]  # accumulate the log determinant of Jacobian
  }

  model <- tensorflow::tf$keras$Model(inputs = inputs, outputs = list(x, total_log_det_jacobian))
  return(model)
}

negative_log_likelihood <- function(y_true, y_pred) {
  z, log_det_jacobian = y_pred 
  logp_z <- -0.5 * tensorflow::tf$reduce_sum(z^2, axis = 1L) - 0.5 * length(z) * log(2 * pi)
  nll = -tensorflow::tf$reduce_mean(logp_z + log_det_jacobian)
  return(nll)
}
                                          

                                          
.train_realnvp <- function(samples, normal_samples, num_coupling_layers = 5, epochs = 50, batch_size = 32, learning_rate = 0.001, train_ratio = 0.8) {
  input_shape <- ncol(samples)
  
  realnvp_model <- .create_realnvp(input_shape, num_coupling_layers)
  
  realnvp_model %>% keras::compile(
    optimizer = keras::optimizer_adam(lr = learning_rate),
    loss = negative_log_likelihood
  )

  history <- realnvp_model %>% keras::fit(
    x = train_samples,
    y = normal_samples,  
    epochs = epochs,
    batch_size = batch_size,
    shuffle = TRUE,
    validation_data = list(test_samples, test_samples)
  )
  
  return(realnvp_model)
}

                                            
                                         
