# This is my file to have:
# 1) basic unit tests and more, and
# 2) high-level generic tests based on synthetic data
# Eventually, it should be split up into multiple files,
# ideally one for each definitions file or even one for
# each function definition and all automated. But for
# now we'll keep it simple.

source("loss-functions.R")
source("gen-synthethic-data.R")
source("gradient-descent.R")
source("stochastic-gradient-descent.R")

# Generate sample data
library(e1071)
library(caret)
library(gettingtothebottom)


################################################
# Test gen-synthetic-data.R
################################################

### TEST for synthetic data with both real-valued and binary labels

test.gen.synthetic.data <- function()
{
  ### TESTS: gen.synthetic.data.continous.labels
  n <- 2
  theta <- c(1, -2)
  data <- gen.synthetic.data.continous.labels(n, theta, seed = 0)
  data
  sigmoid.loss.fn(theta, data)
  model <- gradient.descent(data)
  model$best$theta
  sigmoid.loss.fn(model$best$theta, data)
  
  ### TESTS: gen.synthetic.data.binary.labels
  n <- 2
  theta <- c(1, -2)
  (data <- gen.synthetic.data.binary.labels(2, theta = c(1, -2), seed = 0))
  sigmoid.loss.fn(theta, data)
  model <- gradient.descent(data)
  df.results <- data.frame(maxit=model$params$maxit, theta.1=model$best$theta[[1]],
                           theta.2=model$best$theta[[2]], loss=model$best$loss)
  model <- gradient.descent(data, maxit=1000)
  df.results <- rbind(df.results, data.frame(maxit=model$params$maxit, theta.1=model$best$theta[[1]],
                                             theta.2=model$best$theta[[2]], loss=model$best$loss))
  model <- gradient.descent(data, maxit=2000)
  df.results <- rbind(df.results, data.frame(maxit=model$params$maxit, theta.1=model$best$theta[[1]],
                                             theta.2=model$best$theta[[2]], loss=model$best$loss))
  model <- gradient.descent(data, maxit=5000)
  df.results <- rbind(df.results, data.frame(maxit=model$params$maxit, theta.1=model$best$theta[[1]],
                                             theta.2=model$best$theta[[2]], loss=model$best$loss))
  df.results
}

################################################
### Test gradient-descent.R
################################################

test.gradient.descent <- function()
{
  ### Test Algorithm 1.1 (deterministic ordered batch sampling)
  ### Just testing output of different verbose levels
  theta.1 <- c(1,0,-1)
  data.bin.1 <- gen.synthetic.data.binary.labels(50, theta.1, seed = 0)
  loss.bin.1 <- sigmoid.loss.fn(theta.1, data.bin.1)
  model <- gradient.descent(data.bin.1, B=15, maxit=50, verbose="silent")
  print(cbind(model$df.results, Target=c(theta.1, loss.bin.1, NA)))
  model <- gradient.descent(data.bin.1, B=15, maxit=50, verbose="quiet")
  print(cbind(model$df.results, Target=c(theta.1, loss.bin.1, NA)))
  model <- gradient.descent(data.bin.1, B=15, maxit=50, verbose="verbose")
  print(cbind(model$df.results, Target=c(theta.1, loss.bin.1, NA)))
  model <- gradient.descent(data.bin.1, B=15, maxit=50, verbose="debug")
  print(cbind(model$df.results, Target=c(theta.1, loss.bin.1, NA)))
  
  ### Test Algorithm 1.2 (uniformly random batch sampling without replacement)
  model <- gradient.descent(data.bin.1, B=15, maxit=50, verbose="debug", sample.method="random")
  print(round(cbind(Target=c(theta.1, loss.bin.1, NA), model$df.results), 8))
  
  ### Test Algorithm 1.1 and Algorithm 1.2 each on the same datasets with real-valued labels
  data.cont.1 <- gen.synthetic.data.continous.labels(50, theta.1, seed = 0)
  loss.cont.1 <- sigmoid.loss.fn(theta.1, data.cont.1)
  model <- gradient.descent(data.cont.1, B=15, maxit=50, verbose="quiet")
  print(round(cbind(Target=c(theta.1, loss.cont.1, NA), model$df.results), 8))
  model <- gradient.descent(data.cont.1, B=15, maxit=50, verbose="quiet", sample.method="random")
  print(round(cbind(Target=c(theta.1, loss.cont.1, NA), model$df.results), 8))
  
  theta.2 <- c(1, 2, 3)
  data.cont.2 <- gen.synthetic.data.continous.labels(1000, theta.2, seed = 0)
  loss.cont.2 <- sigmoid.loss.fn(theta.2, data.cont.2)
  model <- gradient.descent(data.cont.2, B=100, maxit=200, verbose="quiet")
  print(round(cbind(Target=c(theta.2, loss.cont.2, NA), model$df.results), 8))
  model <- gradient.descent(data.cont.2, B=100, maxit=200, verbose="quiet", sample.method="random")
  print(round(cbind(Target=c(theta.2, loss.cont.2, NA), model$df.results), 8))
  
  theta.3 <- c(5, -2, 10, 0.5, -1.5)
  data.cont.3 <- gen.synthetic.data.continous.labels(1000, theta.3, mean=2, sd=5, seed = 0)
  loss.cont.3 <- sigmoid.loss.fn(theta.3, data.cont.3)
  model <- gradient.descent(data.cont.3, B=200, maxit=1500, verbose="quiet")
  print(round(cbind(Target=c(theta.3, loss.cont.3, NA), model$df.results), 8))
  model <- gradient.descent(data.cont.3, B=200, maxit=1500, verbose="quiet", sample.method="random")
  print(round(cbind(Target=c(theta.3, loss.cont.3, NA), model$df.results), 8))
}

################################################
### Test stochastic-gradient-descent.R
################################################

test.stochastic.gradient.descent <- function()
{
  ### Test Algorithm 2: Stochastic Gradient Descent, with binary labels 
  ## Testing different verbose levels
  theta.1 <- c(1, 0, -1)
  data.bin.1 <- gen.synthetic.data.binary.labels(50, theta.1, seed = 0)
  loss.bin.1 <- sigmoid.loss.fn(theta.1, data.bin.1)
  model <- stochastic.gradient.descent (data.bin.1, B=15, maxit=50, verbose="silent")
  print(round(cbind(Target=c(theta.1, loss.bin.1, NA), model$df.results), 8))
  model <- stochastic.gradient.descent (data.bin.1, B=15, maxit=50, verbose="quiet")
  print(round(cbind(Target=c(theta.1, loss.bin.1, NA), model$df.results), 8))
  model <- stochastic.gradient.descent (data.bin.1, B=15, maxit=50, verbose="verbose")
  print(round(cbind(Target=c(theta.1, loss.bin.1, NA), model$df.results), 8))
  model <- stochastic.gradient.descent (data.bin.1, B=15, maxit=50, verbose="debug")
  print(round(cbind(Target=c(theta.1, loss.bin.1, NA), model$df.results), 8))
  
  ## Testing Algorithm 2 with a simple case with binary labels
  theta.2 <- c(-1)
  data.bin.2 <- gen.synthetic.data.binary.labels(10, theta.2, seed = 0)
  loss.bin.2 <- sigmoid.loss.fn(theta.2, data.bin.2)
  model <- stochastic.gradient.descent(data.bin.2, maxit=50)
  print(round(cbind(Target=c(theta.2, loss.bin.2, NA), model$df.results), 8))
  
  
  ### Test Algorithm 2: Stochastic Gradient Descent, with real-valued labels
  theta.1 <- c(1, 0, -1)
  data.cont.1 <- gen.synthetic.data.continous.labels(50, theta.1, seed = 0)
  loss.cont.1 <- sigmoid.loss.fn(theta.1, data.cont.1)
  model <- stochastic.gradient.descent(data.cont.1, B=15, maxit=50, verbose="quiet")
  print(round(cbind(Target=c(theta.1, loss.cont.1, NA), model$df.results), 8))
  
  theta.2 <- c(1, 2, 3)
  data.cont.2 <- gen.synthetic.data.continous.labels(1000, theta.2, seed = 0)
  loss.cont.2 <- sigmoid.loss.fn(theta.2, data.cont.2)
  model <- stochastic.gradient.descent(data.cont.2, B=100, maxit=200, verbose="quiet")
  print(round(cbind(Target=c(theta.2, loss.cont.2, NA), model$df.results), 8))
  
  theta.3 <- c(5, -2, 10, 0.5, -1.5)
  data.cont.3 <- gen.synthetic.data.continous.labels(1000, theta.3, mean=2, sd=5, seed = 0)
  loss.cont.3 <- sigmoid.loss.fn(theta.3, data.cont.3)
  model <- stochastic.gradient.descent(data.cont.3, B=200, maxit=1500, verbose="quiet")
  print(round(cbind(Target=c(theta.3, loss.cont.3, NA), model$df.results), 8))
}

test.pathological.cases <- function()
{
  ### Test the pathological case of optimizing over data points that have
  ### discrete/binary labels.
  ### As you can see, the value of theta[1] found by the model
  ### goes towards +infinity. The more steps we use, the bigger it gets.
  theta <- c(1)
  data <- gen.synthetic.data.binary.labels(1, theta, mean = 1, sd = 0)
  loss.bin <- sigmoid.loss.fn(theta, data)
  ## 100 steps
  model <- gradient.descent(data, maxit = 100)
  print(model$best$theta[[1]])
  ## 1000 steps
  model <- gradient.descent(data, maxit = 1000)
  print(model$best$theta[[1]])
  ## 5000 steps
  model <- gradient.descent(data, maxit = 5000)
  print(model$best$theta[[1]])
}


