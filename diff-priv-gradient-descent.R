source("loss-functions.R")

diff.private.gradient.descent <- function
(
  data,
  epsilon, # Privacy param epsilon
  delta, # Privacy param delta
  L, #Lipschitz constant of 'fn.loss'
  fn.loss = sigmoid.loss.fn, # Function must be Lipschitz
  fn.grad = grad.sigmoid.loss.fn,
  maxit = 1000,
  B = 50,
  eta0 = 2,
  verbose = TRUE
)
{
  stop("Not implemented yet.")
}
