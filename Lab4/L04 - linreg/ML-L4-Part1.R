####################################################################
# Machine Learning - MIRI
# Lluís A. Belanche

# LAB 4: Linear regression and beyond
# version of March 2019
####################################################################


set.seed(2222)
par(mfrow=c(1, 1))

####################################################################
# Example 1. Solution of a linear system with linear regression
####################################################################

## First we create a simple data set:

## t = f(x) + epsilon
## where f(x) = (1 + 1/9)(x-1) + 10 (this is the unknown target function = the regression function)
## and epsilon ~ N(0,1) (this introduces an stochastic dependence between x and t)

N <- 10

(X <- matrix(c(rep(1,N), seq(N)),nrow=N))

(t <- seq(10,20,length.out=N) + rnorm(N))

plot(X[,2],t,lwd=3)

#############################################
## 1. Solution via the pseudo-inverse 
#############################################

## (if need be, check this code against your understanding of the lecture slides)

## always take your time to understand ...

# solution of least-squares problems of the form
#       min_w || t - Xw ||^2

(C <- t(X) %*% X)                   # X^T X

(X.pseudo <- solve(C) %*% t(X))       # (X^T X)^{-1} X^T

## this should be the identity matrix (thus we obtain a left pseudo-inverse of X)
X.pseudo %*% X

## this is the solution (the coefficient vector)
(w <- X.pseudo %*% t)

# you can compare this with the truth: slope of 1 + 1/9 and offset 10

# so this is our model ...
lines (X[,2], w[2,1]*X[,2]+w[1,1], type="l")

#############################################
## 2. Solution via the SVD
#############################################

(s <- svd(X))

# the two columns of X are linearly independent <==> rank(X) = 2 <==> we get two singular values different from 0
# in numbers, rank(X) = 2 = min(10,2), hence X is "full rank"

# Now we check that X = U D V^T

D <- diag(s$d)
s$u %*% D %*% t(s$v) # this should be equal to X

# Application to the solution of least-squares problems of the form
#       min_w || t - Xw ||^2

D <- diag(1/s$d)
(w.svd <- s$v %*% D %*% t(s$u) %*% t)

# w.svd should be equal to w
all.equal(w,w.svd)

## Now we use R's very powerful glm() method to perform linear regression by least squares
## We specify numerical regression by choosing the family = gaussian

(sample <- data.frame(x=X,t=t))

# Note that glm() always adds an intercept (a constant regressor equal to 1) *by default*
# So we have two options:

# 1. turn this off (the "-1" in the formula below) and use our own column of 1's
model1 <- glm (t ~ x.2 + x.1 - 1, data=sample, family = gaussian)

# 2. use this built-in feature and ignore our own column of 1's (recommended)
model2 <- glm (t ~ x.2, data=sample, family = gaussian)

# your coefficients (the w vector)
model1$coefficients
model2$coefficients

# Other fields in the glm() return value will be explained below

#############################################
# 3. Why the SVD?

## Why do we prefer the SVD method to direct pseudo-inversion, if both deliver the same results?

## ... because in forming the X^T X matrix some information may be lost

eps <- 1e-3
(X.eps <- matrix(c(1,eps,0,1,0,eps),nrow=3))

(C.eps <- t(X.eps) %*% X.eps)

# this is going to break down ...
eps <- 1e-10
(X.eps <- matrix(c(1,eps,0,1,0,eps),nrow=3))

(C.eps <- t(X.eps) %*% X.eps)

solve(C.eps) 
## raises an error, because the 2x2 "all-ones" matrix is singular
## (the determinant is 1·1 - 1·1 = 0)

## but this is not the right matrix, we simply lost the epsilon along the way
## because of lack of numerical precision in forming t(X.eps) %*% X.eps

## Intermission ... the condition number of a matrix gives an indication of 
## the accuracy of the result of a matrix inversion
## Values near 1 indicate a well-conditioned matrix, the inversion of which is a very
## reliable process (but large values suggest there is trouble ahead)

## The condition number of a matrix is the product of the norm of the matrix and the norm of the inverse
## Using the standard 2-norm, the condition number is the ratio between largest and the smallest (non-zero)
## singular values of the matrix

## The condition number of the matrix X^T X is the square of that of X

X

kappa(X, exact=TRUE)

kappa(t(X) %*% X, exact=TRUE)

## that wasn't really high, but keep reading ...

## Let's see another example:

## an innocent-looking matrix
(A <- matrix(c(rep(1,N), 100+seq(N)),nrow=N))

kappa(A, exact=TRUE)

kappa(t(A) %*% A, exact=TRUE)

## A simple workaround is to center the second column:

A[,2] <- A[,2] - mean(A[,2])

A

kappa(A, exact=TRUE)

kappa(t(A) %*% A, exact=TRUE)

## Now we could solve for the centered matrix and modify the solution to make it
## correspond to that of the original system

## A little exercise: what is the relationship between the two linear systems? In other words, how can we get the 
## solution to the original one with that of the centered one?

# Note: There is a routine that calculates directly the pseudo-inverse (it does so via the SVD):

library(MASS) # ginv()

ginv(A)

####################################################################
# Example 2. Illustration of ridge regression on synthetic data
####################################################################

## Maybe you recall --from the Lecture 1-- the following ideas:

## How can we avoid overfitting? (most often the real danger is in overfitting; this
## is because many ML methods tend to be very flexible, i.e., they are able to represent complex models)

## There are several ways to do this:

## 1) Get more training data (typically out of our grasp)
## 2) Use (that is, sacrifice!) part of the data for validation
## 3) Use an explicit complexity control (aka regularization)

## We already covered the first two.

## Now we are going to use polynomials again to see the effect of regularization

set.seed (7)

N <- 20
N.test <- 1000
a <- 0
b <- 1
sigma.square <- 0.3^2

# Generation of a training sample of size N

x <- seq(a,b,length.out=N)
t <- sin(2*pi*x) + rnorm(N, mean=0, sd=sqrt(sigma.square))
sample <- data.frame(x=x,t=t)

plot(x,t, lwd=3, ylim = c(-1.1, 1.1))
curve (sin(2*pi*x), a, b, add=TRUE, ylim = c(-1.1, 1.1), col="blue")
abline(0,0, lty=2)

# we begin with polynomials of order 1
model <- glm(t ~ x, data = sample, family = gaussian)
prediction <- predict(model)
abline(model, col="red")
( mean.square.error <- sum((t - prediction)^2)/N )

# alternatively, glm() delivers the deviance = sum of square errors
( mean.square.error <- model$deviance/N )

# we prefer to convert it to normalized root MSE or NRMSE
(norm.root.mse <- sqrt(model$deviance/((N-1)*var(t))))

# we continue with polynomials of order 2 (we are creating basis functions!):
# phi_0(x) = 1, phi_1(x) = x, phi_2(x) = x^2

# ... for which we compute the coefficients w_0, w_1, w_2 using a linear method
# and we get the model y(x;w) = w_0 + w_1·phi_1(x) + w_2·phi_2(x)
# as seen in class

model <- glm(t ~ poly(x, 2, raw=TRUE), data = sample, family = gaussian)

summary(model)

# glm() calls w_0 the Intercept, "poly(input, 2, raw = TRUE)1" is phi_1(x), and so on ...

# the coefficients of the polynomial (of the model) are:

model$coefficients

# so our model is
# y(x;w) = 0.6805 -0.4208·x -0.9854·x^2

# let's plot it in red
plot(x,t, lwd=3, ylim = c(-1.1, 1.1))
curve (sin(2*pi*x), a, b, add=TRUE, ylim = c(-1.1, 1.1), col="blue")
abline(0,0, lty=2)
points(x, predict(model), type="l", col="red", lwd=2)

# get the training normalized root MSE (note it is a bit smaller, as reasonably expected)
( norm.root.mse <- sqrt(model$deviance/((N-1)*var(t))) )

## Let's create now a large test sample, for future use
x.test <- seq(a,b,length.out=N.test)
t.test <- sin(2*pi*x.test) + rnorm(N.test, mean=0, sd=sqrt(sigma.square))
test.sample <- data.frame(x=x.test,t=t.test)
plot(test.sample$x, test.sample$t)

######################################
# Right, let's do linear regression on polynomials (a.k.a. polynomial regression),
# from degrees 1 to N-1

p <- 1
q <- N-1

coef <- model <- vector("list",q-p+1)
norm.root.mse.train <- norm.root.mse.test <- rep(0,q-p+1)

for (i in p:q)
{
  model[[i]] <- glm(t ~ poly(x, i, raw=TRUE), data = sample, family = gaussian)
  
  # store the model coefficients, as well as training and test errors
  
  coef[[i]] <- model[[i]]$coefficients
  norm.root.mse.train[i] <- sqrt(model[[i]]$deviance/N)
  
  predictions <- predict (model[[i]], newdata=test.sample)  
  norm.root.mse.test[i] <- sqrt(sum((test.sample$t - predictions)^2)/((N.test-1)*var(test.sample$t)))
}

# we gather everything together

results <- cbind (Degree=p:q, Coefficients=coef, NRMSE.train=norm.root.mse.train, NRMSE.test=norm.root.mse.test)

## we could do plots on the different predictions for the test set, 
## but we already did this on the previous session

## this time we are going to plot the numerical results

plot(results[,1],results[,1], ylim = c(0, 1.1), col="white", xlab="Degree",ylab="errors")
axis(1, at=p:q)
points(x=results[,1],y=results[,3], type="l", col="red", lwd=2)
points(x=results[,1],y=results[,4], type="l", col="blue", lwd=2, add=TRUE)

legend ("topleft", legend=c("TRAINING ERROR","TEST ERROR"),    
        lty=c(1,1), # gives the legend appropriate symbols (lines)
        lwd=c(2.5,2.5), col=c("red","blue")) # gives the legend lines the correct color and width

abline (v=3, lty=2)

# If you get an error message in which R complains ... you can ignore it
# (RStudio's plotting is black magic sometimes)

# What do you see in the plot? try to reflect a little bit

## Last but not least, let's inspect the coefficients for the different degrees

# We will see that all coefficients of the same degree get large (in magnitude)
# as the *maximum* degree grows (except the coefficient of degree 1)

# the column below is the maximum degree of the polynomial
# the row below is the different terms of the polynomial

coefs.table <- matrix (nrow=10, ncol=9)

for (i in 1:10)
  for (j in 1:9)
    coefs.table[i,j] <- coef[[j]][i]

coefs.table

# The conclusion is obvious: we can limit the effective complexity by
# preventing this growth ---> this is what regularization does
# (instead of limiting the maximum degree, we limit the coefficients of all terms)

##########################################################################
# Example 3: Real data modelling with standard, ridge and LASSO regression
##########################################################################

# The following dataset is from a study by Stamey et al. (1989) about prostate cancer, 
# measuring the correlation between the level of a prostate-specific antigen and some covariates:

# lcavol  : log-cancer volume
# lweight : log-prostate weight
# age     : age of patient
# lbhp    : log-amount of benign hyperplasia
# svi     : seminal vesicle invasion
# lcp     : log-capsular penetration
# gleason : Gleason Score, check http://en.wikipedia.org/wiki/Gleason_Grading_System
# pgg45   : percent of Gleason scores 4 or 5
#
# lpsa is the response variable, in logarithms (log-psa)

pcancer <- read.table("prostate.data", header=TRUE)
summary(pcancer)

# There's a training sub-dataset that we will focus on. Later, we will try to predict
# the values of the remaining observations (test)

# Scale data and prepare train/test split
pcancer.std <- data.frame(cbind(scale(pcancer[,1:8]),lpsa=pcancer$lpsa))

train <- pcancer.std[pcancer$train,]
test <- pcancer.std[!pcancer$train,]

dim(train)
dim(test)

# The data looks like this

plot(train)

# Given that this is a biological dataset, some covariates are correlated

round(cor(train),2)

##############################
# LINEAR REGRESSION (standard)

N <- nrow(train)

(model.linreg <- lm(lpsa ~ ., data=train))

## which we simplify using the AIC (not seen in class): the fact is that we can get rid of some variables, useless for the model

# final model and its coeficients (betas):
(model.linreg.FINAL <- step(model.linreg))
beta.linreg.FINAL <- coef(model.linreg.FINAL)

###########################
# LINEAR REGRESSION (RIDGE)

library(MASS)

model.ridge <- lm.ridge(lpsa ~ ., data=train, lambda = seq(0,10,0.1))

plot(seq(0,10,0.1), model.ridge$GCV, main="GCV of Ridge Regression", type="l", 
     xlab=expression(lambda), ylab="GCV")

# The optimal lambda is given by
(lambda.ridge <- seq(0,10,0.1)[which.min(model.ridge$GCV)])
abline (v=lambda.ridge,lty=2)

# We can plot the coefficients and see how they vary as a function of lambda:
colors <- rainbow(8)

matplot (seq(0,10,0.1), coef(model.ridge)[,-1], xlim=c(0,11), type="l",xlab=expression(lambda), 
        ylab=expression(hat(beta)), col=colors, lty=1, lwd=2, main="Ridge coefficients")
abline (v=lambda.ridge, lty=2)
abline (h=0, lty=2)
arrows (5.5,0.45,5,0.35, length = 0.15)
text (rep(10, 9), coef(model.ridge)[length(seq(0,10,0.1)),-1], colnames(train)[-9], pos=4, col=colors)
text(5.5, 0.4, adj=c(0,-1), "best lambda", col="black", cex=0.75)

## So we refit our final ridge regression model using the best lambda:
model.ridgereg.FINAL <- lm.ridge(lpsa ~ ., data=train, lambda = lambda.ridge)
(beta.ridgereg.FINAL <- coef(model.ridgereg.FINAL))

###########################
# LINEAR REGRESSION (LASSO)

## In the LASSO the coefficients are penalized by the L1 norm. The 
# optimal value for lambda is again chosen by cross-validation

library(glmnet)

t <- as.numeric(train[,'lpsa'])
x <- as.matrix(train[,1:8])

model.lasso <- cv.glmnet (x, t, nfolds = 10)
plot(model.lasso)

# The '.' correspond to removed variables
coef(model.lasso)

# lambda.min is the value of lambda that gives minimum mean cross-validated error
model.lasso$lambda.min

# Predictions can be made based on the fitted cv.glmnet object; for instance, this would be the TRAINING error with the "optimal" lambda as chosen by LOOCV
predict (model.lasso, newx = x, s = "lambda.min")

##########################################################
## We can now compare the predictions on the test dataset:
##########################################################

# To make a fair comparison between the three techniques, 
# we should use the same cross-validation for each problem. 
# Standard and ridge regression allow to compute the LOOCV rather fast, but for the LASSO this is not possible. 
# To select the best model, we now use 10x10-CV in all the methods, 
# using the best hyperparameters (lambdas) found previously:

library(caret)

## specify 10x10 CV
K <- 10; TIMES <- 10
trc <- trainControl (method="repeatedcv", number=K, repeats=TIMES)

# STANDARD REGRESSION
model.std.10x10CV <- train (lpsa ~ ., data = train, trControl=trc, method='lm')

normalization.train <- (length(train$lpsa)-1)*var(train$lpsa)
NMSE.std.train <- crossprod(predict (model.std.10x10CV) - train$lpsa) / normalization.train

# RIDGE REGRESSION
# lm.ridge is not in caret's built-in library (sadly)
# this means we need to program everything ourselves (buf!):

data <- train # by just in case

res <- replicate (TIMES, {
  # shuffle the data
  data <- data[sample(nrow(data)),]
  # Create K equally sized folds
  folds <- cut (1:nrow(data), breaks=K, labels=FALSE)
  sse <- 0
  # Perform 10 fold cross validation
  for (i in 1:K)
  {
    valid.indexes <- which (folds==i,arr.ind=TRUE)
    valid.data <- data[valid.indexes, ]
    train.data <- data[-valid.indexes, ]
    model.ridgereg <- lm.ridge (lpsa ~ ., data=train.data, lambda = lambda.ridge)
    beta.ridgereg <- coef (model.ridgereg)
    sse <- sse + crossprod (train[valid.indexes,'lpsa'] - beta.ridgereg[1] 
                            - as.matrix(valid.data[,1:8]) %*% beta.ridgereg[2:9])
  }
  sse/K
})

NMSE.ridge.train <- mean(res) / normalization.train

# LASSO REGRESSION
library(elasticnet)
model.lasso.10x10CV <- train (lpsa ~ ., data = train, trControl=trc, method='glmnet')

NMSE.lasso.train <- crossprod(predict (model.lasso.10x10CV) - train$lpsa) / normalization.train

# Choose the best model as the one with the lowest CV error:
NMSE.std.train; NMSE.ridge.train; NMSE.lasso.train

# Which turns out to be ridge regression

## This is the test NMSE:
normalization.test <- (length(test$lpsa)-1)*var(test$lpsa)

sse <- crossprod (test$lpsa - beta.ridgereg.FINAL[1] 
           - as.matrix(test[,1:8]) %*% beta.ridgereg.FINAL[2:9])
(NMSE.ridge <- sse/normalization.test)

# Final comments:

# 1) In this lab you can see clearly how the volume of work increases when one
# wants to do a thorough analysis, even using a high-level specialized language like R.

# 2) The final model -ridge regression- is not bad but leaves margin for improvement; the difference
# between train (CV) and test errors is large. This is mainly due to the fact that
# the test set is very small, and thus subject to a great variance.

# 3) It is a good exercise to get and compare the final coefficient vectors.
