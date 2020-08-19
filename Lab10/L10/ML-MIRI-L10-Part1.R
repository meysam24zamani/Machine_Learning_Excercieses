####################################################################
# Machine Learning - MIRI Master
# Lluís A. Belanche

# LAB 10: Radial Basis Function Network (Part 1)
# version of April 2019
####################################################################

library(MASS)
library(cclust)

####################################################################
## Radial Basis Function Network Example: regression of a 1D function
####################################################################

set.seed (4)

## We are going to do all the computations "by hand"

## Let us depart from the following function in the (a,b) interval

myf <- function (x) { (1 + x - 2*x^2) * exp(-x^2) }

## We are going to model this function in the interval (-5,5)

a <- -5
b <- 5
sigma.2 <- 0.04
domain <- c(a,b)

myf.data <- function (N, a, b) 
{
  x <- runif(N, a, b)
  t <- myf(x) + rnorm(N, sd=sqrt(sigma.2))
  dd <- data.frame(x,t)
  names(dd) <- c("x", "t")
  dd
}

N <- 100
d <- myf.data (N, a , b)

summary(d)

## The black points are the data, the blue line is the true underlying function

plot (d)
curve (myf, a, b, col='blue', add=TRUE)

## Create a large test data too for future use; notice that the generation mechanism is the same

N.test <- 2000
d.test <- myf.data (N.test, a , b)

# Function to compute a PHI (N x M) design matrix, without the Phi_0(x) = 1 column;
# c.i, sp.i are the centers and smoothing parameters or variances (sigma_i^2) of the neurons, respectively

PHI <- function (x,c,sp)
{
  N <- length(x)
  M <- length(c)
  phis <- matrix(rep(0,M*N), nrow=M)
  for (i in 1:M)
    phis[i,] <- exp(-(x - c[i])^2/(2*sp[i]))
  
  t(phis)
}

## We find the centers and variances for each neuron using k-means; since this clustering algorithm is non-deterministic (because the initial centers are random), we do it 'NumKmeans' times

NumKmeans <- 20

## We set a rather large number of hidden units (= basis functions) M as a function of data size (the sqrt is just a heuristic!) because we are going to try different regularizers

(M <- floor(sqrt(N)))
  
m <- matrix(0,nrow=NumKmeans,ncol=M)
h <- matrix(0,nrow=NumKmeans,ncol=M)

data.Kmeans <- cbind(d$x,rep(0,N))
  
for (j in 1:NumKmeans)
{
    # Find the centers c.i with k-means
    km.res <- cclust (x=data.Kmeans, centers=M, iter.max=200, method="kmeans", dist="euclidean")
    m[j,] <- km.res$centers[,1]
    
    # Obtain the variances sp.i as a function of the c.i
    h[j,] <- rep(0,M)
    for (i in 1:M)
    {
      indexes <- which(km.res$cluster == i)
      h[j,i] <- sum(abs(d$x[indexes] - m[j,i]))/length(indexes)
      if (h[j,i] == 0) h[j,i] <- 1
    }
}
  

## Now for each k-means we get the hidden-to-output weights by solving a regularized
## least-squares problem (standard ridge regression), very much as we did in previous labs

## The difference is that now we perform ridge regression on the PHI matrix (that is, on the new regressors given by the basis functions), not on the original inputs ...

## ... and find the best lambda with using GCV across all choices of basis functions (the NumKmeans clusterings)

(lambda.list <- 10^seq(-3,1.5,by=0.1))

library(MASS) # we need it for lm.ridge

errors <- rep(0,NumKmeans)
bestLambdas <- rep(0,NumKmeans)

# For each k-means' result
for (num in 1:NumKmeans)
{
  c.i <- m[num,]
  sp.i <- h[num,]
  
  myPHI <- PHI (d$x,c.i,sp.i)
  aux1 <- lm.ridge(d$t ~ myPHI, d, lambda = lambda.list)
  my.lambda <- as.numeric(names(which.min(aux1$GCV)))

  aux2 <- lm.ridge(d$t ~ myPHI, d, lambda = my.lambda)
  
  errors[num] <- sqrt(aux2$GCV)
  bestLambdas[num] <- my.lambda
}


## Now we obtain the best model among the tested ones

(bestIndex <- which(errors == min(errors)))
(bestLambda <- bestLambdas[bestIndex])
c.i <- m[bestIndex,]
sp.i <- h[bestIndex,]

## we see that this problem needs a lot of regularization! This makes sense if you take a look at how the data is generated (the previous plot): the noise level is very high relative to the signal
bestLambda

## We also see that the best lambda fluctuates (since the data changes due to the clustering), but the order of magnitude is quite stable)
bestLambdas

## We now create the final model:

my.RBF <- lm.ridge (d$t ~ PHI (d$x,c.i,sp.i), d, lambda = bestLambda)

## these are the final hidden-to-output weights: note how small they are (here is where we regularize)
(w.i <- setNames(coef(my.RBF), paste0("w_", 0:M)))

## It remains to calculate the prediction on the test data

test.PHI <- cbind(rep(1,length(d.test$x)),PHI(d.test$x,c.i,sp.i))
y <- test.PHI %*% w.i

## And now the normalized error of this prediction

(errorsTest <- sqrt(sum((d.test$t - y)^2)/((N.test-1)*var(d.test$t))))

## Much better if we plot everything

par(mfrow=c(1,1))

## Test data in black
plot(d.test$x,d.test$t,xlab="x",ylab="t",main=paste("Prediction (learning size: ",toString(N),"examples)"),ylim=c(-1.5,1.5))

## Red data are the predictions
points(d.test$x,y,col='red',lwd=1)

## and the blue line is the underlying function
curve (myf, a, b, col='blue', add=TRUE)

## The classical (predictive) R^2 coefficient is:
1-errorsTest^2

## Since we know the data generating mechanism, we can compute how good our model is wrt to the best possible model: this model (the regression function), would have a true genearlization error of sigma^2.
## 
## we can therefore compute the so-called "percentage error":
## 

(perc.error <- ( sum((d.test$t - y)^2)/N.test - sigma.2 ) / sigma.2 * 100)

## which means that our model is just 9.3% away from the best possible model

## The previous code is designed for 1D problems but you can easily adapt it to more input dimensions

## There is a general package for neural networks: {RSNNS}

## Which is actually the R interface to the (formerly widely) used and flexible Stuttgart Neural Network Simulator (SNNS)

## This library contains many standard implementations of neural networks. The package actually wraps the SNNS functionality to make it available from within R

## The RBF version within this package has a sophisticated method for initializing the network, which is also quite non-standard, so we avoid further explanation

## Sadly this package does not provide with a way to control regularization or to allow for multi-class problems (a softmax option with the cross-entropy error), so I would not recommend it
## 
## A second package for neural networks in R is {neural}:
## This package trains a RBFNN with backpropagation, so is a rather different view

library(neural)

# For comparison, we use the same number of centers (M)
data <- rbftrain (as.matrix(d$x),M,as.matrix(d$t), visual = FALSE)

# And make it predict the same test data
preds <- rbf (as.matrix(d.test$x),data$weight,data$dist,data$neurons,data$sigma)

## And now the normalized error of this prediction

(errorsTest <- sqrt(sum((d.test$t - preds)^2)/((N.test-1)*var(d.test$t))))

## Much better if we plot everything

par(mfrow=c(1,1))

## Test data in black
plot(d.test$x,d.test$t,xlab="x",ylab="t",main=paste("Prediction (learning size: ",toString(N),"examples)"),ylim=c(-1.5,1.5))

## Red data are the predictions
points(d.test$x,preds,col='red',lwd=1)

## and the blue line is the underlying function
curve (myf, a, b, col='blue', add=TRUE)

## The results are poorer, with an advantage for the former method (using clustering)
1-errorsTest^2
