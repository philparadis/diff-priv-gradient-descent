# This function is used to generate synthethic data based on an underlying
# gene
# Inputs:
#   n = number of points
#   theta = parameters

gen.synthetic.data.binary.labels <- function(n, theta, mean = 0, sd = 1, seed = NULL)
{
  if(!is.null(seed)) set.seed(seed)
  dim <- length(theta)
  features <- matrix(rnorm(n * dim, mean, sd), nrow = dim, ncol = n)
  labels <- (sigmoid(colSums(features * theta)) >= 0.5) * 1
  data <- rbind(features, labels)
  data
}

gen.synthetic.data.continous.labels <- function(n, theta, mean = 0, sd = 1, seed = NULL)
{
  if(!is.null(seed)) set.seed(seed)
  dim <- length(theta)
  features <- matrix(rnorm(n * dim, mean, sd), nrow = dim, ncol = n)
  labels <- sigmoid(colSums(features * theta))
  data <- rbind(features, labels)
  data
}


