# Algorithm 2: Stochastic Gradient Descent

source("loss-functions.R")

# Helper function used when verbose = "debug"
print.debug.stochastic.grad.info <- function(t, maxit, loss, dim, grad, rand.Z, delta, print.as.matrix = FALSE)
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
    print(cbind(grad, rand.Z, delta))
  } else {
    cat(paste0("  grad   = c(", paste0(signif(grad[sub], 3), collapse=", "), extra, ")\n",
               "  rand.Z = c(", paste0(signif(rand.Z[sub], 3), collapse=", "), extra, ")\n",
               "  delta  = c(", paste0(signif(delta[sub], 3), collapse=", "), extra, ")\n"))
  }
}

VERBOSE.DEBUG <- 0
VERBOSE.VERBOSE <- 1
VERBOSE.QUIET <- 2
VERBOSE.SILENT <- 3

### TODO:
### The default values of epsilon, delta and especially L were chosen
### completely randomly and are probably way wrong, bad and horrible.
### Need to think a bit about how to choose good default values.

stochastic.gradient.descent <- function
(
  data,
  epsilon = 0.05, # Privacy param epsilon
  delta   = 0.5, # Privacy param delta
  L       = 10, #Lipschitz constant of 'fn.loss'
  fn.loss = sigmoid.loss.fn, # Function *must* be Lipschitz
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
  sample.method <- "random"
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
    cat(paste0("epsilon:               ", epsilon, "\n"))
    cat(paste0("delta:                 ", delta, "\n"))
    cat(paste0("Lipschitz constant L:  ", L, "\n"))
    cat(paste0("Max num steps:         ", maxit, "\n"))
    cat(paste0("Batch size:            ", B, "\n"))
    cat(paste0("Initial learning rate: ", eta0, "\n"))
    cat(paste0("Verbose:               ", verbose, "\n"))
  } else if (verbose.level <= VERBOSE.QUIET) {
    cat("Parameters:\n")
    cat(paste0("D = ", dim, ", n = ", n, ", maxit = ", maxit, ", B = ", B,
               ", eta0 = ", eta0, ", epsilon = ", epsilon, ", delta = ", delta,
               ", L = ", L, "\nsample.method = ", sample.method, "\n"))
  }
  
  ### 1. Set the privacy parameters (epsilon and delta)
  ## Solve the quadratic equation:
  ##   ax^2 + bx + c = 0
  ## where
  ##   a = 4*T*(B/n)^2
  ##   b = (B/n)*sqrt(8*T*ln(2/delta))
  ##   c = -epsilon
  ## Find x = (-b +/- sqrt(b^2 - 4*a*c))/(2*a)
  ## Then let epsilon.0 <- x
  
  a <- 4*T*(B/n)^2
  b <- (B/n)*sqrt(8*T*log(2/delta))
  c <- -epsilon
  Discrim <- b^2 - 4*a*c
  tol <- 1e-5
  if (Discrim >= tol) {
    # The discriminant is positive, so we have two roots
    SqDiscrim <- sqrt(Discrim)
    root.1 <- (-b + SqDiscrim) / (2*a)
    root.2 <- (-b - SqDiscrim) / (2*a)
    
    root.check.1 <- root.1/(2*log(2/delta))
    root.check.2 <- root.2/(2*log(2/delta))
    
    if (verbose.level <= VERBOSE.VERBOSE) {
      cat("Found 2 roots of the epsilon.0 quadratic equation, giving two possible values for epsilon.0.\n")
      cat(paste0("  root.1 = ", root.1, " root.check.1 = ", root.check.1, "\n",
                 "  root.2 = ", root.2, " root.check.2 = ", root.check.2, "\n"))
    }
  
    if (root.1 < 0 && root.2 >= 0) {
      if (verbose.level <= VERBOSE.VERBOSE) {
        cat(paste0("Since root.1 < 0 and root.2 >= 0, we need to set epsilon.0 to root.2\n"))
      }
      epsilon.0 <- root.2
      check.0 <- root.check.2
    } else if (root.1 >= 0 && root.2 < 0) {
      epsilon.0 <- root.1
      check.0 <- root.check.1
    } else if (root.1 >= 0 && root.2 >= 0) {
      stop("TODO: We have two valid epsilon.0 roots. Not sure which one to select yet.")
      # if (root.check.2 < root.check.1) {
      #   root.1 <- root.2
      #   root.check.1 <- root.check.2
      #   if (verbose.level <= VERBOSE.VERBOSE) {
      #     cat("Selecting root.2 as epsilon.0\n")
      #   }
      # } else {
      #   if (verbose.level <= VERBOSE.VERBOSE) {
      #     cat("Selecting root.1 as epsilon.0\n")
      #   }
      # }
    } else {
      stop("ERROR: Both epsilon.0 roots are negative. Please check your input parameters.")
    }
    epsilon.0 <- root.1
    check.0 <- root.check.1
  } else if (abs(Discrim) < tol) {
    # Assume the discriminant is zero, so we have a single root
    epsilon.0 <- -b / (2*a)
    check.0 <- epsilon.0/(2*log(2/delta))
    
    if (verbose.level <= VERBOSE.DEBUG) {
      cat("Found a single root of the epsilon.0 quadratic equation, giving a single possible value.\n")
    }
  } else {
    # Assume the discriminant is negative, so we have complex roots
    stop("ERROR: The solutions for epsilon.0 are complex-valued numbers. We haven't implemented this case. TODO: Does this case even make any sense?")
  }
  
  if (check.0 > 1) {
    stop(paste0("ERROR: The constrain on \"epsilon.0/(2*log(2/delta)) <= 1",
                "\" could not be satisfied as it equaled: ", check.0))
  }
  
  ### 2. Set sigma.squared (i.e. the variance for the normal distribution that
  ###    model the coordinates of the random variable added as noise to the
  ###    gradient in step 4 (c))
  
  sigma.squared <- (2*L/B)^2 * 2 * log((5/4)*(2/delta))/epsilon.0
  if (sigma.squared < 0) {
    stop(paste0("ERROR: variance = sigma^2 = ", sigma.squared, ", which is negative.\n",
                "TODO: Decide what to do here:\n",
                "    A) Set variance to 0,\n",
                "    B) Take the absolute value of the variance,\n",
                "    C) Figure out what is wrong from the value(s) of epsilon, delta or L.\n",
                "    D) Other ideas? :)\n"))
  }
                
  # In practice, we only need the standard deviation, so we compute it
  # only once now.
  std.dev <- sqrt(abs(sigma.squared))
  
  if (verbose.level <= VERBOSE.VERBOSE) {
    cat("Calculating privacy parameters...")
    cat(paste0("epsilon.0 = ", epsilon.0, "\n"))
  }
  if (verbose.level <= VERBOSE.DEBUG) {
    cat(paste0("Constraint epsilon.0/(2*log(2/delta)) = ", check.0, " <= 1\n"))
  }
  if (verbose.level <= VERBOSE.QUIET) {
    cat(paste0("Privacy parameter: sigma^2 = ", sigma.squared,
               " (equivalently, standard deviation = ", std.dev, ")\n"))
  }

  ### 3. Initialization of initial theta arbitrarily
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
  
  ### 4. Main loop
  for (t in 1:maxit)
  {
    ## (a) This is Algorithm 2, so we pick a random subset of B distinct
    ## elements out of the dataset
    indices.j <- sample(n, B, replace = FALSE)

    if (verbose.level <= VERBOSE.DEBUG) {
      cat(paste0("Step, ", format(t, width=nchar(maxit)), ": Batch indices (j_1 to j_", B, ") = ", 
                 paste0(indices.j, collapse = ", "), "\n"))
    }
    
    ## (b) For now, leave learning rate constant
    ## learning.rate <- eta0
    
    ## (c) Update theta vector.
    ## First compute the gradient of the loss function.
    ## Then, compute a random vector whose coordinates are sampled independently
    ## from the normal distribution N(0, sigma.squared)
    grad <- fn.grad(theta, data, indices.j)
    rand.Z <- rnorm(dim, mean = 0, sd = std.dev)
    delta <- -1 * learning.rate * ((1/B) * grad + rand.Z)
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
        # Print full grad, rand.Z and delta vectors... can get messy...
        print.debug.stochastic.grad.info(t, maxit, loss, dim, grad, rand.Z, delta, TRUE)
      } else if (verbose.level <= VERBOSE.VERBOSE) {
        # Print step #, current loss and average grad and delta values
        print.debug.stochastic.grad.info(t, maxit, loss, dim, grad, rand.Z, delta, FALSE)
      } else if (verbose.level <= VERBOSE.QUIET) {
        cat(paste0("Step ", format(t, width=nchar(maxit)), ": ",loss,
                   " (||grad|| = ", signif(sqrt(sum(grad^2)), 6),
                   ", ||delta|| = ", signif(sqrt(sum(delta^2)), 6),
                   ", ||rand.Z|| = ", signif(sqrt(sum(rand.Z^2)), 6), ")\n"))
      } else {
        # Just print step # and current loss
        cat(paste0("Step ", format(t, width=nchar(maxit)), ": loss = ", loss, "\n"))
      }
    }
  }
  
  ### 5. Final phase. Set last step theta and loss.
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
                   privacy.params = list(epsilon = epsilon,
                                         delta = delta,
                                         L = L,
                                         epsilon.0 = epsilon.0,
                                         sigma.squared = sigma.squared,
                                         std.dev = std.dev),
                   df.results = df.results)
  #class(my.model) <- "stochastic.gradient.descent"
  return(my.model)
}

print.stochastic.gradient.descent <- function(model, ...)
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

summary.stochastic.gradient.descent <- function(model, ...)
{
  structure(model, class="summary.stochastic.gradient.descent")
}
