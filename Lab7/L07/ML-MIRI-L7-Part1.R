####################################################################
# Machine Learning - MIRI Master
# Lluís A. Belanche

# LAB 7: GLMs - Logistic Regression and beyond (Part1)
# version of April 2019
####################################################################

####################################################################
# Starter: Maximum Likelihood (ML)
####################################################################

set.seed(1234)

## First we are going to play with a very simple example of Maximum Likelihood for the binomial distribution
## The goal is to estimate the probability p of getting heads in N coin tosses if we get n1 heads (exactly as seen in class)

## Suppose we toss a coin N=100 times and we get n1=70 heads

N <- 100
n1 <- 70

p <- seq(from=1e-5, to=1, by=0.01)

# this is the likelihood function (as a function of p); note it tops at 0.7
L <- choose(N,n1)*p^n1*(1-p)^(N-n1)

plot(p,L,type="l",ylab="likelihood of p",xaxt = "n")
grid(nx=10)
axis(side = 1, at = seq(0,1,by=0.1), las = 2, hadj = 0.9)

# this is the log-likelihood function (as a function of p); it also tops at 0.7
# note log() is the natural logarithm in R

logL <- log(choose(N,n1)) + n1*log(p) + (N-n1)*log(1-p)

plot(p,logL,type="l",ylab="log-likelihood of p",xaxt = "n")
grid(nx=10)
axis(side = 1, at = seq(0,1,by=0.1), las = 2, hadj = 0.9)

# the maximum is attained at n1/N = 0.7

####################################################################
## Example 1: Logistic Regression with artificial data
####################################################################

## The purpose of this first example is to get acquainted with the basics of logistic regression and the call to glm()

## glm() is used to fit Generalized Linear Models

# You may need to recall at this point the logistic regression model from the slides

## Let x represent a single continuous predictor
# Let t represent a class ('0' or '1'), with a probability p of being 1 for every x, that is related linearly to the predictor via the logit funtion

# That is: logit(p) = beta_1*x + beta_0

# Therefore logit() is the link function

N <- 500
x <- rnorm(n=N, mean=3, sd=2)     # generate the x_n data (note x is a vector)
beta_1 <- 0.6 ; beta_0 <- -1.5    # this is the ground truth, which is unknown

p <- 1/(1+exp( -(beta_1*x + beta_0) ))  # generate the p_n (note p is a vector)

t <- rbinom (n=N,size=1,prob=p)    # generate the targets (classes) according to p
t <- as.factor(t)                  # note t is a vector

# just for plotting, we convert t to number
plot(x,as.numeric(t)-1)

glm.res <- glm (t~x, family = binomial(link=logit)) 
# actually link=logit is the default for binomial

## look at the coefficients!
# 'Intercept' is beta_0, 'x' is beta_1

summary(glm.res)

## Obviously x is very significant (and the Intercept is almost always significant)

## Therefore, our estimated model is

# logit(p_n) = 0.67693*x_n - 1.61652

# quite close to the ground truth!

## In general (I mean, in the multivariate case) you get this as:

coef(glm.res)

# or, if you want individual coefficients:

glm.res$coefficients["x"]

glm.res$coefficients["(Intercept)"]

## Interpretation of the coefficients:

# For a 1 unit increase in x, there is an increase in the odds for t by a factor of:

exp(glm.res$coefficients["x"])

# In plain words, the odds for t increases by a 96.8% for a 1 unit increase in x

## That is almost doubling the odds in this case

# let us try now a "logistic plot" using predict()
M <- max(x)
m <- min(x)

# We are going to generate abscissae in which to compute the model:

# We divide the interval (m,M) in 200 equally-spaced parts
# then add 10 points before and 10 after at the same distance

abscissae <- m + (-10:210)*(M-m)/200 

# with type="response" we get the predicted probability
preds <- predict (glm.res, data.frame(x=abscissae),type="response")

# Now plot the prediction together with data points

plot(p~x,ylim=c(0,1)) # plot previous data

lines(abscissae, preds, col="blue") # add our model, quite good!

####################################################################
## Example 2: Poisson Regression with artificial data
####################################################################

## Now let us switch to Poisson regression
# You may need to recall at this point the Poisson regression model from the slides
# (recall log() is now the link function)

# Let x represent the distance to workplace (in km)
# Let t represent the amount of time wasted in travelling from home to work and back (in hours per week)
# (let us suppose this is measured as an integer quantity)

# (0 means "I work at home, but sometimes I have to go to another place")

## We now generate artificial "realistic" data from a Poisson process:

N <- 500
x <- runif(n=N,0.1,12)            # generate the x_n (note x is a vector)
beta_1 <- 0.35 ; beta_0 <- -1     # this is the ground truth, which is unknown
l <- exp( beta_1*x+beta_0 )       # generate the lambda (note l is a vector)
t <- rpois (n=N, lambda = l)       # generate the targets t according to l

plot(x,t,xlab="Distance to workplace (km)", ylab="Time wasted (h/week)")

# Fitting of Poisson regression
mydata <- data.frame(h.week=t, dist=x)

glm.res <- glm(h.week ~ dist, family = poisson(link="log"), data = mydata)

summary(glm.res)

## Therefore, our estimated model is log(l_n) = 0.338*dist_n - 0.89
## (fairly close to the ground truth)

## Interpretation of the coefficients:

## For a 1 unit increase in distance (that is, 1 km), the expected number of hours wasted in travelling from home to work and back increases by a factor of:

exp(0.337965)

## a 40.2% on average

## We can finally gather everything together and plot it:

(new.d <- seq(0,30,length.out=100))
fv <- predict (glm.res, data.frame(dist=new.d),se=TRUE)

plot (x,t,xlab="Distance to workplace (km)", ylab="Time wasted (h/week)")
lines (new.d,exp(fv$fit), col='red')

## We can even add a confidence interval at the 95%level

lines (new.d,exp(fv$fit+1.967*fv$se.fit), col='red',lty=2)
lines (new.d,exp(fv$fit-1.967*fv$se.fit), col='red',lty=2)

## Note that the model is not very precise, but this is because the uncertainty in the problem (and therefore in the generated data) is very high. The model is actually (almost) as good as it can be.


####################################################################
## Example 3: Logistic Regression from the book by Venables & Ripley
####################################################################

## The purpose of this example is to illustrate the power of GLM fitting. We are going to model again a logistic regression with a call to glm(). 

# This time, however, our simulated data comes in a different form: the goal is to model the fraction of some species of insects killed at different doses of pyrethroids 
# (Pyrethroids are a chemical class of active ingredients found in many of the modern insecticides)

## Note 20 insects are being "used" in all treatments.

# logarithms (in base 2) of used doses, repeated twice (for males and for females)
(logdose <- rep(0:5,2))

# number of dead insects at different doses, for each sex
(numdead <- c(1,4,9,13,18,20,0,2,6,10,12,16))

(sex <- factor(rep(c("M","F"),each=6)))

## Our little experiment

(budworm <- data.frame (logdose,numdead,numalive=20-numdead,sex))

## We now do logistic regression to predict the fraction of dead insects:

# We declare the number of "trials" (the fixed 20 insects) with the 'weights' parameter
bud.logreg <- glm (numdead/20 ~ logdose + sex, family=binomial, data=budworm, weights=rep(20,length(logdose)))

# Since the number of insects is constant (20) in all trials, we can use an equivalent command, with a simpler syntax
bud.logreg <- glm(cbind(numdead,numalive) ~ logdose + sex, family=binomial, data=budworm)


# in this model, the effect of sex is significant, though less than the log of the dose
summary(bud.logreg)

# now show data and model

# plots x axis in logarithmic scale  (because of logdose)
plot(c(1,32),c(0,1),type="n",xlab="dose",ylab="non-survival probabiblity",log="x")

# plot the data
text(2^logdose,numdead/20,labels=as.character(sex),col=as.integer(sex))

grid <- 1:32

# compute the model for males on grid, using "predict" on the values with log_2(grid)
model.on.males <- predict(bud.logreg, data.frame(logdose=log2(grid),sex="M"),type="response",se=TRUE)

# plot the model
lines(grid,model.on.males$fit,col="red")

lines (grid,model.on.males$fit+1.967*model.on.males$se.fit, col='red',lty=2)
lines (grid,model.on.males$fit-1.967*model.on.males$se.fit, col='red',lty=2)

# the same with sex = "F"
model.on.females <- predict(bud.logreg, data.frame(logdose=log2(grid),sex="F"),type="response",se=TRUE)

lines(grid,model.on.females$fit,col="black")

lines (grid,model.on.females$fit+1.967*model.on.females$se.fit, col='black',lty=2)
lines (grid,model.on.females$fit-1.967*model.on.females$se.fit, col='black',lty=2)

# The plot shows the model works quite fine in modelling the death probability
# as a function of logdose (and sex); may be it is a bit better for males (males are more predictable)
# It also shows that the treatment affects differently males and females (the latter being more resistant)

####################################################################
## Example 4: Logistic regression for spam mail
####################################################################

## This example will illustrate how to change the 'cut point' for prediction, when there is an interest in minimizing a particular source of errors

## We recall here our example of spam mail prediction from a previous lab

library(kernlab)  

data(spam)

## We redo our preprocessing

spam[,55:57] <- as.matrix(log10(spam[,55:57]+1))

spam2 <- spam[spam$george==0,]
spam2 <- spam2[spam2$num650==0,]
spam2 <- spam2[spam2$hp==0,]
spam2 <- spam2[spam2$hpl==0,]

george.vars <- 25:28
spam2 <- spam2[,-george.vars]

moneys.vars <- c(16,17,20,24)
spam3 <- data.frame( spam2[,-moneys.vars], spam2[,16]+spam2[,17]+spam2[,20]+spam2[,24])

colnames(spam3)[51] <- "about.money"

dim(spam3)

set.seed (4321)
N <- nrow(spam3)                                                                                              
learn <- sample(1:N, round(0.67*N))
nlearn <- length(learn)
ntest <- N - nlearn

## Previously we had obtained a predictive (test) error of 26.26% using Naive Bayes in this same partition of the data

## Fit a GLM in the learning data
spamM1 <- glm (type ~ ., data=spam3[learn,], family=binomial)


# (do not worry about the warnings: they are fitted probabilities numerically very close to 0 or 1)

## Simplify it using the AIC (this may take a while, since there are many variables)
# (this takes a while)
spamM1.AIC <- step (spamM1)

## We define now a convenience function:

# 'P' is a parameter; whenever our filter assigns spam with probability at least P then we predict spam
spam.accs <- function (P=0.5)
{
  ## Compute accuracy in learning data
  
  spamM1.AICpred <- NULL
  spamM1.AICpred[spamM1.AIC$fitted.values<P] <- 0
  spamM1.AICpred[spamM1.AIC$fitted.values>=P] <- 1
  
  spamM1.AICpred <- factor(spamM1.AICpred, labels=c("nonspam","spam"))
  
  print(M1.TRtable <- table(Truth=spam3[learn,]$type,Pred=spamM1.AICpred))
  
  print(100*(1-sum(diag(M1.TRtable))/nlearn))
   
  ## Compute accuracy in test data
  
  gl1t <- predict(spamM1.AIC, newdata=spam3[-learn,],type="response")
  gl1predt <- NULL
  gl1predt[gl1t<P] <- 0
  gl1predt[gl1t>=P] <- 1
  
  gl1predt <- factor(gl1predt, labels=c("nonspam","spam"))
  
  print(M1.TEtable <- table(Truth=spam3[-learn,]$type,Pred=gl1predt))
  
  print(100*(1-sum(diag(M1.TEtable))/ntest))
}

spam.accs()
# gives 7.21% TRAINING ERROR and 7.07% TESTING ERROR

## Although the errors are quite low (and much better than those with Naive Bayes), still one 
# could argue that we should try to lower the probability of predicting spam when it is not
# We can do this (at the expense of increasing the converse probability) by:

spam.accs(0.7)

# gives 9.66% TRAINING ERROR and 10.3% TESTING ERROR

## So we get a much better spam filter; notice that the filter has a very low probability of 
## predicting spam when it is not (which is the delicate case)

####################################################################
## Example 5: Logistic regression to model heart disease
####################################################################

## This last example of logistic regression will connect to the example in the slides and will show
## that we can improve our models by incorporating non-linear predictors.

## The enzyme creatinine kynase (CK) has been suggested as an aid for early diagnosis of heart attack

## We copy data from Hand et al '94 (360 patients)

# CK values
(CK <- seq(20,460,by=40))

# number of patients suffering a heart attack
(num.heart <- c(2,13,30,30,21,19,18,13,19,15,7,8))

# number of patients not suffering a heart attack
(num.non.heart <- c(88,26,8,5,0,1,1,1,1,0,0,0))

## Our experimental data

(heart.attack <- data.frame (CK,num.heart,num.non.heart))

## Plot the data

p <- heart.attack$num.heart/(heart.attack$num.heart+heart.attack$num.non.heart)

plot (heart.attack$CK, p, xlab="CK level", ylab="Proportion of heart attack")

## Let's try our first model (this is similar to the previous example with insects)

mod.1 <- glm(cbind(num.heart,num.non.heart) ~ CK, family=binomial, data=heart.attack)

summary(mod.1)

## This asks for a residuals versus fitted values plot
plot(mod.1, which=1)

## This plot shows a clear non-linear trend (possibly cubic); another signal of bad fit is the
# residual deviance (36.929). This quantity is a sum of squares and theoretically distributed 
# (look at the summary) as a Chi^2 with 10 degrees of freedom. If we do:

1-pchisq(36.929,10)

# which is the (in this case very small) probability for a Chi^2_10 random variable being as large as 36.929

## The residual plots suggest trying a cubic predictor:

mod.2 <- glm(cbind(num.heart,num.non.heart) ~ CK + I(CK^2) + I(CK^3), family=binomial, data=heart.attack)

summary(mod.2)

## We can see by the deviance that the model is much better. The value of 4.2525 is not too large to be 
# consistent with a Chi^2_8 random variable:

1-pchisq(4.2525,8)

plot(mod.2, which=1)

## The AIC has also improved substantially. The residual plots are also much flatter (and more constant), except in two cases

## We can now display our new cubic model:

cks <- seq(10,500,length.out=100)
  
has <- predict (mod.2, data.frame(CK=cks),se=TRUE, type="response")

plot (heart.attack$CK, p, xlab="CK level", ylab="Proportion of heart attack")

lines(cks,has$fit,col="red")

lines (cks,has$fit+1.967*has$se.fit, col='red',lty=2)
lines (cks,has$fit-1.967*has$se.fit, col='red',lty=2)

