####################################################################
# Machine Learning - MIRI Master
# Lluís A. Belanche

# LAB 11: Kernel methods - the SVM for classification (Part 1)
# version of May 2019
####################################################################


set.seed(6046)


####################################################################
# Modelling artificial 2D sinusoidal data for two-class problems
####################################################################


## the SVM is located in two different packages: one of them is 'e1071'
library(e1071)

## First we create a simple two-class data set:

N <- 200

make.sinusoidals <- function(m,noise=0.2) 
{
  x1 <- c(1:2*m)
  x2 <- c(1:2*m)
  
  for (i in 1:m) {
    x1[i] <- (i/m) * pi
    x2[i] <- sin(x1[i]) + rnorm(1,0,noise)
  }
  
  for (j in 1:m) {
    x1[m+j] <- (j/m + 1/2) * pi
    x2[m+j] <- cos(x1[m+j]) + rnorm(1,0,noise)
  }
  
  target <- as.factor(c(rep(+1,m),rep(-1,m)))
  
  return(data.frame(x1,x2,target))
}

## let's generate the data
dataset <- make.sinusoidals (N)

## and have a look at it
summary(dataset)

plot(dataset$x1,dataset$x2,col=dataset$target)

## Now we wish to fit and visualize different SVM models

## model 1: LINEAR kernel, C=1 (cost parameter)
(model <- svm(dataset[,1:2],dataset[,3], type="C-classification", cost=1, kernel="linear", scale = FALSE))

## Now we are going to visualize what we have done; since we have artificial data, instead of creating a random test set, we can create a grid of points as test

source("plot-prediction.R")

## plot the data, the OSH with margins, the support vectors (marked with a surrounding box), ...
plot.prediction (model, "linear kernel, C=1")

## model 2: linear kernel, C=0.1 (cost parameter)
(model <- svm(dataset[,1:2],dataset[,3], type="C-classification", cost=0.1, kernel="linear", scale = FALSE))

## the margin is wider (lower VC dimension), the number of support vectors is larger (more margin violations)
plot.prediction (model, "linear kernel, C=0.1")

## model 3: linear kernel, C=25 (cost parameter)
(model <- svm(dataset[,1:2],dataset[,3], type="C-classification", cost=25, kernel="linear", scale = FALSE))

## the margin is narrower (higher VC dimension), number of support vectors is smaller (less violations of the margin)
plot.prediction (model, "linear kernel, C=25")

## Let's put everything together, for 6 values of C:

par(mfrow=c(2,3))

for (C in 10^seq(-2,3))
{
  model <- svm(dataset[,1:2],dataset[,3], type="C-classification", cost=C, kernel="linear", scale = FALSE)
  plot.prediction (model, paste ("linear kernel (C=", C, ") ", model$tot.nSV, " Support Vectors", sep=""))
}


## Now we move to a QUADRATIC kernel (polynomial of degree 2); the kernel has the form:
## k(x,y) = (<x,y> + coef0)^degree

## quadratic kernel, C=1 (cost parameter)
(model <- svm(dataset[,1:2],dataset[,3], type="C-classification", cost=1, kernel="polynomial", degree=2, coef0=1, scale = FALSE))

par(mfrow=c(1,1))

## notice that neither the OSH or the margins are linear (they are quadratic); they are linear in the feature space
## in the previous linear kernel, both spaces coincide
plot.prediction (model, "quadratic kernel, C=1")


## Let's put it together directly, for 6 values of C:

par(mfrow=c(2,3))

for (C in 10^seq(-2,3))
{
  model <- svm(dataset[,1:2],dataset[,3], type="C-classification", cost=C, kernel="polynomial", degree=2, coef0=1, scale = FALSE)
  plot.prediction (model, paste ("quadratic kernel (C=", C, ") ", model$tot.nSV, " Support Vectors", sep=""))
}

## Now we move to a CUBIC kernel (polynomial of degree 3); the kernel has the form:
## k(x,y) = (<x,y> + 1)^3

## cubic kernel, C=1 (cost parameter)
(model <- svm(dataset[,1:2],dataset[,3], type="C-classification", cost=1, kernel="polynomial", degree=3, coef0=1, scale = FALSE))

par(mfrow=c(1,1))

## neither the OSH or the margins are linear (they are now cubic); they are linear in the feature space
## this choice seems much better, given the structure of the classes
plot.prediction (model, "cubic kernel, C=1")

## Let's put it together directly, for 6 values of C:

par(mfrow=c(2,3))

for (C in 10^seq(-2,3))
{
  model <- svm(dataset[,1:2],dataset[,3], type="C-classification", cost=C, kernel="polynomial", degree=3, coef0=1, scale = FALSE)
  plot.prediction (model, paste ("cubic kernel (C=", C, ") ", model$tot.nSV, " Support Vectors", sep=""))
}

## Finally we use the Gaussian RBF kernel (polynomial of infinite degree; the kernel has the form:
## k(x,y) = exp(-gamma·||x - y||^2), gamma>0

## RBF kernel, C=1 (cost parameter)
(model <- svm(dataset[,1:2],dataset[,3], type="C-classification", cost=1, kernel="radial", scale = FALSE))

## the default value for gamma is 0.5

par(mfrow=c(1,1))

plot.prediction (model, "radial kernel, C=1, gamma=0.5")

## Let's put it together directly, for 6 values of C, holding gamma constant = 0.5:

par(mfrow=c(2,3))

for (C in 10^seq(-2,3))
{
  model <- svm(dataset[,1:2],dataset[,3], type="C-classification", cost=C, kernel="radial", scale = FALSE)
  plot.prediction (model, paste ("RBF kernel (C=", C, ") ", model$tot.nSV, " Support Vectors", sep=""))
}

## Now for 8 values of gamma, holding C constant = 1:

par(mfrow=c(2,4))

for (g in 2^seq(-3,4))
{
  model <- svm(dataset[,1:2],dataset[,3], type="C-classification", cost=1, kernel="radial", gamma=g, scale = FALSE)
  plot.prediction (model, paste ("RBF kernel (gamma=", g, ") ", model$tot.nSV, " Support Vectors", sep=""))
}

## In practice we should optimize both (C,gamma) at the same time

## How? Using cross-validation or trying to get "good" estimates analyzing the data

## Now we define a utility function for performing k-fold CV:

## the learning data is split into k equal sized parts
## every time, one part goes for validation and k-1 go for building the model (training)
## the final error is the mean prediction error in the validation parts
## Note k=N corresponds to LOOCV

## a typical choice is k=10
k <- 10 
folds <- sample(rep(1:k, length=N), N, replace=FALSE) 

valid.error <- rep(0,k)


## This function is not intended to be useful for general training purposes, but it is useful for the sake of illustration
## in particular, it does not optimize the value of C (it requires it as parameter)

train.svm.kCV <- function (which.kernel, myC, kCV=10)
{
  for (i in 1:kCV) 
  {  
    train <- dataset[folds!=i,] # for building the model (training)
    valid <- dataset[folds==i,] # for prediction (validation)
    
    x_train <- train[,1:2]
    t_train <- train[,3]
    
    switch(which.kernel,
           linear={model <- svm(x_train, t_train, type="C-classification", cost=myC, kernel="linear", scale = FALSE)},
           poly.2={model <- svm(x_train, t_train, type="C-classification", cost=myC, kernel="polynomial", degree=2, coef0=1, scale = FALSE)},
           poly.3={model <- svm(x_train, t_train, type="C-classification", cost=myC, kernel="polynomial", degree=3, coef0=1, scale = FALSE)},
           RBF={model <- svm(x_train, t_train, type="C-classification", cost=myC, kernel="radial", scale = FALSE)},
           stop("Enter one of 'linear', 'poly.2', 'poly.3', 'radial'"))
    
    x_valid <- valid[,1:2]
    pred <- predict(model,x_valid)
    t_true <- valid[,3]
    
    # compute validation error for part 'i'
    valid.error[i] <- sum(pred != t_true)/length(t_true)
  }
  # return average validation error
  100*sum(valid.error)/length(valid.error)
}

# Fit an SVM with linear kernel

C <- 1

(VA.error.linear <- train.svm.kCV ("linear", myC=C))

## The procedure is to choose the model with the lowest CV error and then refit it with the whole learning data,
## then use it to predict the test set; we will do this at the end

## Fit an SVM with quadratic kernel 

(VA.error.poly.2 <- train.svm.kCV ("poly.2", myC=C))

## ## Fit an SVM with cubic kernel

(VA.error.poly.3 <- train.svm.kCV ("poly.3", myC=C))

## we get a series of decreasing CV errors ...

## and finally an RBF Gaussian kernel 

(VA.error.RBF <- train.svm.kCV ("RBF", myC=C))

## Now in a real scenario we should choose the model with the lowest CV error
## which in this case is the RBF (we get a very low CV error because this problem is easy for a SVM)

## so we choose RBF and C=1 and refit the model in the whole training set (no CV)
model <- svm(dataset[,1:2],dataset[,3], type="C-classification", cost=C, kernel="radial", scale = FALSE)

## and make it predict a test set:

## let's generate the test data
dataset.test <- make.sinusoidals (1000)

## and have a look at it
summary(dataset.test)

par(mfrow=c(1,1))
plot(dataset.test$x1,dataset.test$x2,col=dataset.test$target)

pred <- predict(model,dataset.test[,1:2])
t_true <- dataset.test[,3]

table(pred,t_true)

# compute testing error (in %)
(sum(pred != t_true)/length(t_true))

## In a real setting we should also optimize the value of C, again
## with CV; all this can be done very conveniently using tune() to do
## automatic grid-search

## Other packages provide with heuristic methods to estimate the gamma in the RBF kernel (see next labs)
