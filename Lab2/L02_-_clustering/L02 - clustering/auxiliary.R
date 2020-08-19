# GENERATE DATA FROM A MIXTURE OF 2D GAUSSIANS
generate.data <- function(N, K, prior.mean, prior.var)
{
  p <- length(prior.mean)
  
  # generate random mixture centres from the prior
  mu_k <- mvrnorm(K, mu=prior.mean, Sigma=diag(prior.var, 2))
  
  # generate mixture coefficients
  pi_k <- runif(K)
  pi_k <- pi_k/sum(pi_k)
  
  # generate the data
  obs <- matrix(0, nrow=N, ncol=p)
  z <- numeric(N)
  sigma_k <- matrix(0, nrow=K, ncol=p)
  
  for (i in 1:K)
    sigma_k[i,] <- runif(p)
  
  for (i in 1:N)
  {
    # draw the observation from a component according to coefficient
    z[i] <- sample(1:K, 1, prob=pi_k)
    # draw the observation from the corresponding mixture location
    obs[i,] <- mvrnorm(1, mu=mu_k[z[i],], Sigma=diag(sigma_k[z[i],],p))
  }
  list(locs=mu_k, z=z, obs=obs, coefs=pi_k)
}

# plot 2d data from a mixture
plot.mixture <- function(locs, z, obs)
{
  stopifnot(dim(obs)[2]==2)
  z <- as.factor(z)
  df1 <- data.frame(x=obs[,1], y=obs[,2], z=z)
  df2 <- data.frame(x=locs[,1], y=locs[,2])
  p <- ggplot()
  p <- p + geom_point(data=df1, aes(x=x, y=y, colour=z), shape=16, size=2, alpha=0.75)
  p <- p + geom_point(data=df2, aes(x=x, y=y), shape=16, size=3)
  p <- p + theme(legend.position="none")
  p
}

# plot 2d data as a scatter plot
plot.data <- function(dat)
{
  stopifnot(dim(dat)[2]==2)
  df1 <- data.frame(x=dat[,1], y=dat[,2])
  p <- ggplot()
  p <- p + geom_point(data=df1, aes(x=x, y=y), size=2, alpha=0.75)
  p
}
