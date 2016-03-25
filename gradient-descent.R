### This file defines the function 'gradient-descent-, which can handle both
### Algorithm 1.1 (Fully deterministic gradient descent) and Algorithm 1.2
### (Gradient descent with uniformly random batching)
### The Algorithm 1.1 is used by default and Algorithm 1.2 is used by
### passing the argument sample.method = "random" to the function.

source("loss-functions.R")

# Helper function used when verbose = "debug"
print.debug.grad.info <- function(t, maxit, loss, dim, grad, delta, print.as.matrix = FALSE)
{
  # Also print gradient values and delta values
  if (dim <= 5) {
    sub <- 1:dim
    extra <- ""
  } else {
    sub <- 1:5
    extra <- ", ..."
  }
  cat(paste0("Step, ", format(t, width=nchar(maxit)), ": loss = ", loss, "\n"))
  if (print.as.matrix == TRUE) {
    print(cbind(grad, delta))
  } else {
    cat(paste0("  grad   = c(", paste0(signif(grad[sub], 3), collapse=", "), extra, ")\n",
               "  delta  = c(", paste0(signif(delta[sub], 3), collapse=", "), extra, ")\n"))
  }
}

VERBOSE.DEBUG <- 0
VERBOSE.VERBOSE <- 1
VERBOSE.QUIET <- 2
VERBOSE.SILENT <- 3

gradient.descent <- function
(
  data,
  sample.method = c("ordered", "random"),
  fn.loss = sigmoid.loss.fn,
  fn.grad = grad.sigmoid.loss.fn,
  maxit = 1000,
  B = 50,
  eta0 = 2,
  verbose = c("quiet", "verbose", "silent", "debug")
)
{
  # Set parameters
  dim <- nrow(data) - 1
  n <- ncol(data)
  sample.method <- match.arg(sample.method)
  verbose <- match.arg(verbose)
  
  if (verbose == "debug")
    verbose.level <- VERBOSE.DEBUG
  else if (verbose == "verbose")
    verbose.level <- VERBOSE.VERBOSE
  else if (verbose == "quiet")
    verbose.level <- VERBOSE.QUIET
  else
    verbose.level <- VERBOSE.SILENT
  
  # Note: Features are encoded as vectors z_i = data[1:dim, i, drop=FALSE]
  # Whereas labels are encoded as vectors y_i = data[dim + 1, i, drop=FALSE]
  # For all i in  {1, ..., n}
  
  # Alternatively, we can view those as matrices:
  # Features are the matrix Z = data[1:dim, , drop=FALSE]
  
  ### Sanitize parameters
  maxit <- as.integer(maxit)
  B <- as.integer(B)
  if (B > n)
  {
    if (verbose.level <= VERBOSE.QUIET) {
      message(paste0("Batch size B = ", B, ", is larger than the number of ",
                     "data points (", n, "), hence setting B = ", n, "."))
    }
    B <- n
  }
  if (eta0 <= 0) {
    stop("Error: Invalid initial learning rate eta0.")
  }
  
  # Print learning parameters
  if (verbose.level <= VERBOSE.VERBOSE) {
    cat("Parameters:\n")
    cat(paste0("Dimension:             ", dim, "\n"))
    cat(paste0("Num points:            ", n, "\n"))
    cat(paste0("Sample method:         ", sample.method, "\n"))
    cat(paste0("Max num steps:         ", maxit, "\n"))
    cat(paste0("Batch size:            ", B, "\n"))
    cat(paste0("Initial learning rate: ", eta0, "\n"))
    cat(paste0("Verbose:               ", verbose, "\n"))
  } else if (verbose.level <= VERBOSE.QUIET) {
    cat("Parameters:\n")
    cat(paste0("D = ", dim, ", n = ", n, ", maxit = ", maxit, ", B = ", B, ", eta0 = ", eta0,
               ", sample.method = ", sample.method, "\n"))
  }
  
  ### 1. Initialization of initial theta arbitrarily
  theta <- runif(dim, -1, 1)
  learning.rate <- eta0
  
  # Initialize loss variables for bookkeeping
  loss.init <- fn.loss(theta, data)
  loss.init.theta <- theta
  loss.init.step <- 1
  loss.best <- loss.init
  loss.best.theta <- theta
  loss.best.step <- 1
  loss.last <- loss.init
  
  ### 2. Main loop
  for (t in 1:maxit)
  {
    ## (a) Pick a subset of B different data points by specifying a vector of
    ## indices for x_{j_1}, ..., x_{j_B}
    if (sample.method == "ordered") {
      ## This is the method from Algorithm 1.1, so we pick the indices
      ## deterministically, in order.
      if (t == 1) {
        indices.j <- 1:B
      } else {
        # Move the indices forward by a window of size B modulo
        indices.j <- ((indices.j - 1 + B) %% n) + 1
      }
    } else {
      ## This is the method from Algorithm 1.2, so we pick a random subset of
      ## B distinct elements out of the dataset
      indices.j <- sample(n, B, replace = FALSE)
    }
    
    if (verbose.level <= VERBOSE.DEBUG) {
      cat(paste0("Step ", format(t, width=nchar(maxit)), ": Batch indices (j_1 to j_", B, ") = ", 
                 paste0(indices.j, collapse = ", "), "\n"))
    }
    
    ## (b) For now, leave learning rate constant
    ## learning.rate <- eta0
    
    ## (c) Update theta vector
    grad <- fn.grad(theta, data, indices.j)
    delta <- -1 * learning.rate * (1/B) * grad
    theta <- theta + delta
    
    # Compute new global loss
    loss <- fn.loss(theta, data)
    if (loss < loss.best) {
      loss.best <- loss
      loss.best.theta <- theta
      loss.best.step <- t
    }
    # Print only step loss every 10% of the way
    # (Except in verbose mode, print first 20 steps as well)
    if (verbose.level <= VERBOSE.DEBUG ||
        (verbose.level <= VERBOSE.VERBOSE && (t <= 5 || t == maxit || t %% ceiling(maxit/20) == 0)) ||
        (verbose.level <= VERBOSE.QUIET && (t == 1 || t == maxit || t %% ceiling(maxit/10) == 0)))
    {
      if (verbose.level <= VERBOSE.DEBUG) {
        # Print full grad and delta vectors... can get messy...
        print.debug.grad.info(t, maxit, loss, dim, grad, delta)
      } else if (verbose.level <= VERBOSE.VERBOSE) {
        # Print step #, current loss and average grad and delta values
        cat(paste0("Step ", format(t, width=nchar(maxit)), ": ", loss, " (avg. |grad| = ", mean(abs(grad)),
                   ", avg. |delta| = ", mean(abs(delta)), ")\n"))
      } else {
        # Just print step # and current loss
        cat(paste0("Step ", format(t, width=nchar(maxit)), ": loss = ", loss, "\n"))
      }
    }
  }
  
  ### 3. Final phase. Set last step theta and loss.
  loss.last.theta <- theta
  loss.last <- fn.loss(loss.last.theta, data)
  
  if (verbose.level <= VERBOSE.VERBOSE) {
    cat(paste0("Done!... Reached max number of steps, maxit = ", maxit, "\n"))
  }
  
  df.results <- NULL
  df.results <- data.frame(best.theta=loss.best.theta, last.theta=loss.last.theta)
  df.results <- rbind(df.results, c(loss.best, loss.last), c(loss.best.step, maxit))
  dimnames(df.results) <- list(c(paste0("theta[",1:dim,"]"), "loss", "step"),
                               c("Best step", "Final step"))
  df.results <- cbind(df.results, data.frame(initial.step=c(loss.init.theta, loss.init, 1)))
  colnames(df.results)[3] <- "Initial step"
  
  if(verbose.level <= VERBOSE.VERBOSE) {
    cat("Final results:\n")
    print(df.results)
  } else if (verbose.level <= VERBOSE.QUIET) {
    cat("Final results:\n")
    print(df.results[ , 1:2])
  }
  
  # Prepare data structure to output
  my.model <- list(best = list(theta = loss.best.theta,
                               loss = loss.best,
                               step = loss.best.step),
                   last = list(theta = loss.last.theta,
                               loss = loss.last,
                               step = maxit),
                   params = list(dim = dim,
                                 n = n,
                                 sample.method = sample.method,
                                 fn.loss = fn.loss,
                                 gn.grad = fn.grad,
                                 maxit = maxit,
                                 B = B,
                                 eta0 = eta0,
                                 verbose = verbose),
                   df.results = df.results)
  #class(my.model) <- "gradient.descent"
  return(my.model)
}

gradient.descent.deterministic <- function(data, ...)
{
  gradient.descent(data, sample.method = "ordered", ...)
}

gradient.descent.random.batch.sampling <- function(data, ...)
{
  gradient.descent(data, sample.method = "random", ...)
}

print.gradient.descent <- function(model, ...)
{
  cat("Parameters used for Gradient Descent:\n")
  cat(paste0("Dimension:               dim = ", model$params$dim, "\n"))
  cat(paste0("Num points:                n = ", model$params$n, "\n"))
  cat(paste0("Max num of steps:      maxit = ", model$params$maxit, "\n"))
  cat(paste0("Batch size:                B = ", model$params$B, "\n"))
  cat(paste0("Batch sampling method:       = ", model$params$sample.method, "\n"))
  cat(paste0("Initial learning rate:  eta0 = ", model$params$eta0, "\n"))
  cat(paste0("Verbose:                     = ", model$params$verbose, "\n"))
  
  m <- data.frame(best.theta=model$best$theta, last.theta=model$last$theta)
  m <- rbind(m, c(model$best$loss, model$last$loss))
  dim <- model$params$dim
  row.names(m) <- c(paste0("theta[",1:dim,"]"), "loss")
  cat("Solutions and corresponding values of loss function:\n")
  print(m)
}

summary.gradient.descent<- function(model, ...)
{
  structure(model, class="summary.gradient.descent")
}
