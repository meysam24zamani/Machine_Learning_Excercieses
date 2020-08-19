####################################################################
# Machine Learning - MIRI Master
# Lluís A. Belanche

# LAB 13: Random Forests
# version of May 2019
####################################################################

set.seed (6046)

###################################################
## Example 1: Spam mail detection
###################################################

library(kernlab)  

data(spam)

## We do some basic preprocessing

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
summary(spam3)

## We do not need to split the available data into learning and test sets, because the OOB is already a honest estimation of prediction performance

# However, since we want to compare different models (for example, tuning a random forest' hyperparameters), we do it, exactly as we have been doing it so far: we split the available data into learning and test sets, selecting randomly 2/3 and 1/3 of the data

set.seed (4321)

N <- nrow(spam3)                                                                                              
learn <- sample(1:N, round(0.67*N))
nlearn <- length(learn)

## First try a standard classification tree (CART)

library(tree)

model.tree <- tree (type ~ ., data=spam3[learn,])

summary(model.tree)

# so training error rate is 9%

model.tree

plot (model.tree)
text (model.tree,pretty=0)

# let's make it predict the leftout test set

pred.tree <- predict (model.tree, spam3[-learn,], type="class")

(ct <- table(Truth=spam3[-learn,]$type, Pred=pred.tree))

## To better study the results, we define a convenience function (the harmonic mean), to compute the F1 accuracy:

harm <- function (a,b) { 2/(1/a+1/b) }

# percent by class
prop.table(ct, 1)
# total percent correct
sum(diag(ct))/sum(ct)

# test error is 11.21%

round(100*(1-sum(diag(ct))/sum(ct)),2)

# The errors are quite well balanced, although someone could claim that we should make an effort to reduce the nonspam classified as spam:

(F1 <- harm (prop.table(ct,1)[1,1], prop.table(ct,1)[2,2]))


## Now a random Forest
library(randomForest)

model.rf1 <- randomForest(type ~ ., data=spam3[learn,], ntree=100, proximity=FALSE)

model.rf1

## We get an estimated test error (OOB) of 6.82%, and both errors are quite well balanced; let's compute the real test error:

pred.rf1 <- predict (model.rf1, spam3[-learn,], type="class")

(ct <- table(Truth=spam3[-learn,]$type, Pred=pred.rf1))

# percent by class
prop.table(ct, 1)
# total percent correct
sum(diag(ct))/sum(ct)

# real test error is 

round(100*(1-sum(diag(ct))/sum(ct)),2)

(F1 <- harm (prop.table(ct,1)[1,1], prop.table(ct,1)[2,2]))

## So the prediction error of the RF is much better than that of a single tree; 
## however, there is still an issue in the unbalanced errors

## We should try to lower the probability of predicting spam when it is not
# We can do this (at the expense of increasing the converse probability)

# One way to deal with this is to stratify the sampling in the boostrap resamples:

# 'nonspam' is the less represented class, so we upsample it

(N.nonspam <- table(spam3[learn,]$type)["nonspam"])
(N.spam <- table(spam3[learn,]$type)["spam"])

set.seed(1234)

model.rf2 <- randomForest(type ~ ., data=spam3[learn,], ntree=100, proximity=FALSE, 
                          sampsize=c(nonspam=850, spam=500), strata=spam3[learn,]$type)

model.rf2

## We get an estimated test error (OOB) of 7.12%, and now nonspam is better detected; let's compute the real test error:

pred.rf2 <- predict (model.rf2, spam3[-learn,], type="class")

(ct <- table(Truth=spam3[-learn,]$type, Pred=pred.rf2))

# percent by class
prop.table(ct, 1)
# total percent correct
sum(diag(ct))/sum(ct)

# real test error is 

round(100*(1-sum(diag(ct))/sum(ct)),2)

(F1 <- harm (prop.table(ct,1)[1,1], prop.table(ct,1)[2,2]))

## So we get a much better spam filter

## Now we can try to optimize the number of trees, guided by the OOB:

(ntrees <- round(10^seq(1,3,by=0.2)))

# prepare the structure to store the partial results

rf.results <- matrix (rep(0,2*length(ntrees)),nrow=length(ntrees))
colnames (rf.results) <- c("ntrees", "OOB")
rf.results[,"ntrees"] <- ntrees
rf.results[,"OOB"] <- 0

ii <- 1

for (nt in ntrees)
{ 
  print(nt)
  
  model.rf <- randomForest(type ~ ., data=spam3[learn,], ntree=nt, proximity=FALSE, 
                           sampsize=c(nonspam=850, spam=500), strata=spam3[learn,]$type)
  
  # get the OOB
  rf.results[ii,"OOB"] <- model.rf$err.rate[nt,1]
  
  ii <- ii+1
}

rf.results

# choose best value of 'ntrees'

lowest.OOB.error <- as.integer(which.min(rf.results[,"OOB"]))
(ntrees.best <- rf.results[lowest.OOB.error,"ntrees"])

## Now refit the RF with the best value of 'ntrees'

model.rf3 <- randomForest(type ~ ., data=spam3[learn,], ntree=ntrees.best, proximity=FALSE, 
                          sampsize=c(nonspam=850, spam=500), strata=spam3[learn,]$type)

# let's compute the final test error:

pred.rf3 <- predict (model.rf3, spam3[-learn,], type="class")

(ct <- table(Truth=spam3[-learn,]$type, Pred=pred.rf3))

# percent by class
prop.table(ct, 1)
# total percent correct
sum(diag(ct))/sum(ct)

# real test error is 

round(100*(1-sum(diag(ct))/sum(ct)),2)

(F1 <- harm (prop.table(ct,1)[1,1], prop.table(ct,1)[2,2]))

# we can now have a closer look at our model:

summary(model.rf3)
print(model.rf3)

model.rf3$oob.times
model.rf3$confusion

# The importance of variables
importance(model.rf3)
varImpPlot(model.rf3)

## 'about.money' is the most important variable, then 'charDollar', etc

# plot error rate as a function of the number of trees used: 
# black = out of bag (OOB), red = label 1 ('nonspam'), green  = label 2 ('spam')

plot(model.rf3)

legend("topright", legend=c("OOB", "nonspam", "spam"),    
       pch=c(1,1), col=c("black","red","green"))

# What variables are being used in the forest (their total counts)

varUsed(model.rf3, by.tree=FALSE, count = TRUE)

## Data visualization

visual.indexes <- sample(1:N, round(2*N/3))

visual.data <- spam3[visual.indexes,]

rf <- randomForest (type ~ ., data = visual.data, ntree=ntrees.best, proximity=TRUE, importance=FALSE)

print(rf)

## Do Multidimensional Scaling on (1 - proximity); WARNING: this takes some minutes

rf.plot <- MDSplot (rf, visual.data$type,pch=as.numeric(visual.data$type))

library(RColorBrewer)

#add legend and title
title("Data visualization based on MDS projection of the Random Forest")
legend("bottomright", legend=levels(visual.data$type),
       fill=brewer.pal(length(levels(visual.data$type)), "Set1"))

# need to identify some points?
text(rf.plot$points[1:100,],labels=attr(rf.plot$points,"dimnames")[[1]][1:10], cex=0.4) 

####################################################################
# Example 2: classification model for deposit subscription
####################################################################

## Direct marketing campaigns (phone calls) of a Portuguese banking institution. 
## The classification goal is to predict if the client will subscribe a term deposit

set.seed (6046)

deposit <- read.table ("Bank Marketing/bank-full.csv.gz", header=TRUE, stringsAsFactors=TRUE, sep=";")

dim(deposit)

summary(deposit)

## This dataset needs a lot of pre-processing ... it also displays a good mixture of categorical and numeric variables

# age seems OK
# job has 12 values, let's check their frequency

# seems OK
table(deposit$job)

# education has 4 values, let's check their frequency

# seems OK
table(deposit$education)

# month looks very suspicious ... but is OK

table(deposit$month)

# highly skewed ...

hist(deposit$duration)
hist(log(deposit$duration))

deposit$duration <- log(deposit$duration+1)
deposit$previous <- log(deposit$previous+1)

# what to do with 'pdays'? it is not clear how to best pre-process it; we shall need some financial expertise

## The rest seem OK (but it would take a careful analysis, and a lot of domain knowledge)
## In particular, 'duration' is the call duration and maybe it is not fair to include it in the analysis, at least not for prediction purposes

# I rename the target ...

colnames(deposit)[ncol(deposit)] <- "subscribed"

dim(deposit)

summary(deposit)

## Since we want to use different methods, we need CV and a separate test set:

library(TunePareto)

# precalculate the TR/TE partition and the cross-validation partitions on the TR part

N <- nrow(deposit)
all.indexes <- 1:N

learn.indexes <- sample(1:N, round(2*N/3))
test.indexes <- all.indexes[-learn.indexes]

learn.data <- deposit[learn.indexes,]

nlearn <- length(learn.indexes)
ntest <- N - nlearn

## First try a standard classification tree (CART)

library(tree)

model.tree <- tree (subscribed ~ ., data = learn.data)

summary(model.tree)

# so training error rate is 10.72%

model.tree

plot (model.tree)
text (model.tree,pretty=0)

pred.tree <- predict (model.tree, deposit[test.indexes,], type="class")

(ct <- table(Truth=deposit$subscribed[test.indexes], Pred=pred.tree))

## We define now a convenience function (the harmonic mean):

harm <- function (a,b) { 2/(1/a+1/b) }

# percent by class
prop.table(ct, 1)
# total percent correct
sum(diag(ct))/sum(ct)

# test error is 

round(100*(1-sum(diag(ct))/sum(ct)),2)

# not very good, because the 'yes' class is nearly ignored
(F1 <- harm (prop.table(ct,1)[1,1], prop.table(ct,1)[2,2]))

## Now a random Forest
library(randomForest)

model.rf1 <- randomForest(subscribed ~ ., data = learn.data, ntree=100, proximity=FALSE)

model.rf1

## We get an estimated test error (OOB) of 9.3%, so better; let's compute the real test error:

pred.rf1 <- predict (model.rf1, deposit[test.indexes,], type="class")

(ct <- table(Truth=deposit$subscribed[test.indexes], Pred=pred.rf1))

# percent by class
prop.table(ct, 1)
# total percent correct
sum(diag(ct))/sum(ct)

# real test error is 

round(100*(1-sum(diag(ct))/sum(ct)),2)

(F1 <- harm (prop.table(ct,1)[1,1], prop.table(ct,1)[2,2]))

## So OOB really works in estimating prediction error and the RF is better than a single tree

## However, there still is a big issue in unbalanced classes

# one way to deal with this is to include class weights
model.rf2 <- randomForest(subscribed ~ ., data = learn.data, ntree=100, proximity=FALSE, classwt=c(1,10))

model.rf2

# which helps a little bit, but not much: we get estimated test error (OOB) of 9.86% with a better balance; let's compute the real test error:

pred.rf2 <- predict (model.rf2, deposit[test.indexes,], type="class")

(ct <- table(Truth=deposit$subscribed[test.indexes], Pred=pred.rf2))

# percent by class
prop.table(ct, 1)
# total percent correct
sum(diag(ct))/sum(ct)

# real test error is 

round(100*(1-sum(diag(ct))/sum(ct)),2)

(F1 <- harm (prop.table(ct,1)[1,1], prop.table(ct,1)[2,2]))

# another way is to stratify the sampling in the boostrap resamples

# 'yes' is the less represented class, so we upsample it

n.yes <- table(learn.data$subscribed)["yes"]
n.no <- table(learn.data$subscribed)["no"]

model.rf3 <- randomForest(subscribed ~ .-duration, data = learn.data, ntree=100, proximity=FALSE, 
                          sampsize=c(yes=3500, no=3500), strata=learn.data$subscribed)

model.rf3

# which seems to help much more: we get an estimated test error (OOB) of 17%, but with a better balance

# let's compute the real test error:

pred.rf3 <- predict (model.rf3, deposit[test.indexes,], type="class")

(ct <- table(Truth=deposit$subscribed[test.indexes], Pred=pred.rf3))

# percent by class
prop.table(ct, 1)
# total percent correct
sum(diag(ct))/sum(ct)

# real test error is 

round(100*(1-sum(diag(ct))/sum(ct)),2)

(F1 <- harm (prop.table(ct,1)[1,1], prop.table(ct,1)[2,2]))

## Now we can try to optimize the number of trees, guided by OOB:

(ntrees <- round(10^seq(1,3,by=0.2)))

# prepare the structure to store the partial results

rf.results <- matrix (rep(0,2*length(ntrees)),nrow=length(ntrees))
colnames (rf.results) <- c("ntrees", "OOB")
rf.results[,"ntrees"] <- ntrees
rf.results[,"OOB"] <- 0

ii <- 1

for (nt in ntrees)
{ 
  print(nt)

  model.rf <- randomForest(subscribed ~ ., data = learn.data, ntree=nt, proximity=FALSE, 
                          sampsize=c(yes=3000, no=3000), strata=learn.data$subscribed)
  
  # get the OOB
  rf.results[ii,"OOB"] <- model.rf$err.rate[nt,1]

  ii <- ii+1
}

rf.results

# choose best value of 'ntrees'

lowest.OOB.error <- as.integer(which.min(rf.results[,"OOB"]))
(ntrees.best <- rf.results[lowest.OOB.error,"ntrees"])

# we could also try to optimize the number of variables in the same way but, as it was said in the lectures, the default value (square root) works fine in general

## Now refit the RF with the best value of 'ntrees'

model.rf <- randomForest(subscribed ~ ., data = learn.data, ntree=ntrees.best, 
                         sampsize=c(yes=3500, no=3500), strata=learn.data$subscribed,
                         proximity=FALSE, importance=TRUE)

## let's compute the real test error:

pred.rf.final <- predict (model.rf, deposit[test.indexes,], type="class")

(ct <- table(Truth=deposit$subscribed[test.indexes], Pred=pred.rf.final))

# percent by class
prop.table(ct, 1)
# total percent correct
sum(diag(ct))/sum(ct)

# real test error is 

round(100*(1-sum(diag(ct))/sum(ct)),2)

(F1 <- harm (prop.table(ct,1)[1,1], prop.table(ct,1)[2,2]))

print(model.rf)

# The importance of variables
importance(model.rf, type=1)
varImpPlot(model.rf, type=1)

## 'duration' is the most important variable, then 'month', etc

# Plot error rate: 
#   black = out of bag (OOB), 
#   red = just label 1 ('no'), 
#   green  = just label 2 ('yes')
# as a function of the number of trees used

plot(model.rf)

legend("topright", legend=c("OOB", "no", "yes"),    
       pch=c(1,1), col=c("black","red","green"))

# What variables are being used in the forest (their counts)
varUsed(model.rf, by.tree=FALSE,count = TRUE)
