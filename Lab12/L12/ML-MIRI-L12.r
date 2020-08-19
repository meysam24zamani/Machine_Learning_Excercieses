####################################################################
# Machine Learning - MIRI Master
# Lluís A. Belanche

# LAB 12: Kernel methods - the SVM for regression
# version of May 2019
####################################################################

set.seed(555)
library(e1071)

####################################################################
# Example 1. Playing with the SVM for regression and 1D data
####################################################################

## Now we do regression; we have an extra parameter: the 'epsilon', which controls the width of the epsilon-insensitive tube (in feature space)

## a really nice-looking function
A <- 10
a <- pi
f <- function(x) sin(a*x)/(a*x)

# data creation
x <- seq(-A,A,by=0.11)
y <- f(x) + rnorm(x,sd=0.05)
plot(x,y,type="l")

## With this choice of the 'epsilon', 'gamma' and C parameters, the SVM underfits the data (blue line):

model1 <- svm (x,y,epsilon=0.01)
plot(f,-A,A)
lines(x,predict(model1,x),col="blue")

## With this choice of the 'epsilon', 'gamma' and C parameters, the SVM overfits the data (green line):

model2 <- svm (x,y,epsilon=0.01,gamma=200, C=200)
lines(x,predict(model2,x),col="green")

## With this choice of the 'epsilon', 'gamma' and C parameters, the SVM has a decent fit (red line):
model3 <- svm (x,y,epsilon=0.01,gamma=50)
lines(x,predict(model3,x),col="red")

## The other nice package where the SVM is located is {kernlab}

library(kernlab)

## The ksvm() method in this package has some nice features, as creation of user-defined kernels (not seen in this course), built-in cross-validation (via the 'cross' parameter) and automatic estimation of the gamma parameter for the RBF kernel, via the sigest() function (this is transparent to the user).

##########################################################################
# Example 2: Real data modelling with the SVM for regression (revisited)
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
pcancer.std <- data.frame(cbind(scale(pcancer[,1:8]),lpsa=log(pcancer$lpsa)))

train <- pcancer.std[pcancer$train,]
test <- pcancer.std[!pcancer$train,]

dim(train)
dim(test)

# Given that this is a biological dataset, some covariates are correlated

round(cor(train),2)

N <- nrow(train)

(model.QUAD <- ksvm (lpsa ~ ., data=train, type = "eps-svr", epsilon = 1,
                     kernel=polydot(degree = 2, scale = 1, offset = 1),C=1,cross=N))

# sparsity    
nSV(model.QUAD)/N*100

# NMSE
preds <- predict(model.QUAD, newdata=test)
normalization.test <- (length(test$lpsa)-1)*var(test$lpsa)
crossprod(test$lpsa - preds) / normalization.test

(model.RBF <- ksvm (lpsa ~ ., data=train, type = "eps-svr", epsilon = 0.05, C=1,cross=N))

# sparsity    
nSV(model.QUAD)/N*100

# NMSE
preds <- predict(model.RBF, newdata=test)
crossprod(test$lpsa - preds) / normalization.test

plot(test$lpsa,preds)

####################################################################
## Example 3. Analysing a difficult two-class problem with genetic data
####################################################################

## In genetics, a promoter is a region of DNA that facilitates the transcription of a particular gene. 
## Promoters are located near the genes they regulate.

## Promoter Gene Data: data sample that contains DNA sequences, classified into 'promoters' and 'non-promoters'.
## 106 observations, 2 classes (+ and -)
## The 57 explanatory variables describing the DNA sequence have 4 possible values, represented 
## as the nucleotide at each position:
##    [A] adenine, [C] cytosine, [G] guanine, [T] thymine.

## The goal is to develop a predictive model (a classifier) for promoters

## data reading
dd <- read.csv2("promotergene.csv")

d <- ncol(dd)
N <- nrow(dd)

summary(dd)

## Since the data is categorical, we perform a Multiple Correspondence Analysis
## (the analog of PCA or Principal Components Analysis for numerical data)

## you have to source the auxiliary file
source ("acm.r")

X <- dd[,2:d]

mc <- acm(X)

# this is the projection of our data in the first two factors

plot(mc$rs[,1],mc$rs[,2],col=dd[,1])

# Histogram of eigenvalues (we have N-1 of them)

barplot(mc$vaps)

# estimation of the number of dimensions to keep:

i <- 1
while (mc$vaps[i] > mean(mc$vaps)) i <- i+1

(nd <- i-1)

## Create a new dataframe 'Psi' for convenience

Psi <- as.data.frame(cbind(mc$rs[,1:nd], dd[,1]))

names(Psi)[43] <- "Class"
Psi[,43] <- as.factor(Psi[,43])
attach(Psi)

## split data into learn (2/3) and test (1/3) sets
set.seed (2)
learn <- sample(1:N, round(0.67*N))

##### Support vector machine

## The implementation in {kernlab} is a bit more flexible than the one
## in {e1071}: we have the 'cross' parameter, for cross-validation (default is 0)

# we start with a linear kernel

## note how we are going to specify LOOCV (recommended for small datasets only)
mi.svm.1 <- ksvm (Class~.,data=Psi[learn,],kernel='polydot',C=1,cross=length(learn))

# note Number of Support Vectors, Training error and Cross validation error

mi.svm.1

# choose quadratic kernel now
quad <- polydot(degree = 2, scale = 1, offset = 1)

mi.svm.2 <- ksvm (Class~.,data=Psi[learn,],kernel=quad,C=1,cross=length(learn))

# note Number of Support Vectors, Training error and Cross validation error
mi.svm.2

# choose now the RBF kernel with automatic adjustment of the variance
mi.svm.3 <- ksvm (Class~.,data=Psi[learn,],C=1,cross=length(learn))

# note Number of Support Vectors, Training error and Cross validation error
mi.svm.3

# idem but changing the cost parameter C
mi.svm.4 <- ksvm (Class~.,data=Psi[learn,],C=5,cross=length(learn))

# note Number of Support Vectors, Training error and Cross validation error
mi.svm.4

# we choose RBF with C=5 and use it to predict the test set

svmpred <- predict (mi.svm.4, Psi[-learn,-43])

tt <- table(Truth=dd[-learn,1],Pred=svmpred)

(error_rate.test <- 100*(1-sum(diag(tt))/sum(tt)))

# gives a prediction error of 14.3%
