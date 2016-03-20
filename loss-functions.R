library(e1071)
library(microbenchmark)
library(ggplot2)

sanity.check.loss.func.params <- function(theta, data, select.cols)
{
  if (is.null(theta))
    stop("Parameter 'theta' is NULL.")
  if (is.null(data))
    stop("Parameter 'data' is NULL.")
  dim <- length(theta)
  if (!is.matrix(data))
    data <- as.matrix(data)
  if (is.null(select.cols))
    select.cols <- 1:ncol(data)
  n <- length(select.cols)
  if (nrow(data) != dim + 1)
    stop(paste0("Parameter 'data' has wrong dimensions (expected ",
                dim + 1, " x ",  n, "), but instead is ",
                dim(data)[[1]], " x ", dim(data)[[2]], "."))
  if (min(select.cols) < 1 || max(select.cols) > ncol(data))
    stop(paste0("Invalid columns subset (select.cols = ",
                paste0(select.cols, collapse = " "),
                "). There are only ", ncol(data), " columns in 'data'."))
}

### The main standard loss function based on the sigmoid
sigmoid.loss.fn <- function(theta, data, select.cols = NULL)
{
  sanity.check.loss.func.params(theta, data, select.cols)
  if (!is.matrix(data))
    data <- as.matrix(data)
  if (is.null(select.cols))
    select.cols <- 1:ncol(data)
  dim <- length(theta)
  n <- length(select.cols)
  Z <- data[1:dim, select.cols, drop = F]
  y <- data[dim + 1, select.cols, drop = F]
  (1/n) * sum((sigmoid(.colSums(theta * Z, dim, n)) - y) ^ 2)
}

### The (pretty straightforward) gradient of the above loss function
grad.sigmoid.loss.fn <- function(theta, data, select.cols = NULL)
{
  # Note: sigmoid.loss.fn is a function
  #  l : R^n --> R
  # so Grad(l) : R^n --> R^n
  sanity.check.loss.func.params(theta, data, select.cols)
  if (is.vector(data))
    data <- as.matrix(data)
  dim <- length(theta)
  if (is.null(select.cols))
    select.cols <- 1:ncol(data)
  n <- length(select.cols)
  Z <- data[1:dim, select.cols, drop = F]
  y <- data[dim + 1, select.cols, drop = F]
  scalars <- 2 * (sigmoid(.colSums(theta * Z, dim, n)) - y) * dsigmoid(.colSums(theta * Z, dim, n))
  rowSums(matrix(rep(scalars, dim), dim, byrow = TRUE) * Z)
}

run.benchmark.sigmoid.loss.fn <- function()
{
  # Small example data:
  set.seed(1000)
  dim <- 10
  theta <- runif(dim, -5, 5)
  m <- dim + 1
  n <- 20
  select.cols <- 1:n
  data <- matrix(rnorm(m*n), nrow=m, ncol=n)
  Z <- data[1:dim, select.cols, drop = F]
  y <- data[dim + 1, select.cols, drop = F]
  pre.comp.th <- t(matrix(theta))
  th4<- matrix(theta, nrow=1)
  
  # Perform some sanity check
  std.l <- sigmoid.loss.fn(theta, data, select.cols)
  cat(paste0("sigmoid.loss.fn   = ", std.l, "\n"))
  cat(paste0("benchmark #1 loss = ", l1, "\n"))
  cat(paste0("benchmark #2 loss = ", l2, "\n"))
  stopifnot(l1 == std.l)
  
  # Now, run benchmarks of the all four previous sigmoid loss functions
  # as well as the Gradient of the simoid loss function
  res <- microbenchmark(l1 <- (1/n) * sum((sigmoid(.colSums(theta * Z, dim, n)) - y) ^ 2),
                        {th <- t(matrix(theta)); l2 <- (1/n) * sum((sigmoid(th %*% Z) - y) ^ 2)},
                        l3 <- (1/n) * sum((sigmoid(pre.comp.th %*% Z) - y) ^ 2),
                        l4 <- (1/n) * sum((sigmoid(th4 %*% Z) - y) ^ 2),
                        grad1 <- grad.sigmoid.loss.fn(theta, data),
                        times = 1000, control = list(warmup=50))
  autoplot(res)
}

