####################################################################
# Machine Learning - MIRI
# Lluís A. Belanche

# LAB 2: Clustering (k-means and E-M) practise
# version of February 2019
####################################################################

# Choose one of these exercises to practice

####################################################################
## Exercise 1:  Clustering artificial 2-D CIRCLE DATA

library(mlbench)

# We generate 2D data: each of the clusters is a 2-D Gaussian. The centers are equally spaced 
# on a circle around the origin with radius r
# The covariance matrices are of the form sigma^2 I (sd^2 parameter in  mlbench.2dnormals())

N <- 1000
K <- 6

# the clusters
data.1 <- mlbench.2dnormals (N,K)
plot(data.1)

# the raw data (what the clustering method will receive)

plot(x=data.1$x[,1], y=data.1$x[,2])


## Do the following (in separate sections)

# 1. Decide beforehand which clustering method will work best and with which settings.
#    Hint: have a look at the way the data is generated: ?mlbench.2dnormals
# 2. Apply k-means a number of times with fixed K=6 and observe the results
# 3. Apply k-means with a choice of K values of your own and monitor the CH index; which K looks better?
# 4. Apply E-M with K=6 and observe the results (means, coefficients and covariances)
# 5. Check the results against tour expectations (#1.)

####################################################################
## Exercise 2:  Clustering real 2-D data

## This exercise involves the use of the 'Geyser' data set, which contains data from the ‘Old Faithful’ geyser 
## in Yellowstone National Park, Wyoming. 

## the MASS library seems to contain the best version of the data
library(MASS)

help(geyser, package="MASS")

summary(geyser)

plot(geyser)

## with ggplot2, maybe we get better plots:
library(ggplot2)

qplot(waiting, duration, data=geyser)

# 6. Decide beforehand which clustering method will work best and with which settings.
#    No hint this time, this is a real dataset
# 7. Apply k-means with different values of K and observe the results
# 8. Apply k-means 100 times, get averages of the CH index, and decide the best value of K. Does it work?
# 9. Apply E-M with a family of your choice ("spherical", "diagonal", etc), with the best value fo K delivered by k-means
# 10. Choose the model and number of clusters with the largest BIC

# 11. Apply E-M again with a family of your choice ("spherical", "diagonal", etc), this time letting BIC decide the best number of clusters
#     The easiest way to inspect the final results is with summary() of your mixmodCluster() call
# 12. Once you're done, try and plot the results; just plot() the result of mixmodCluster()

####################################################################
## Exercise 3:  Clustering real multi-dimensional data

## This exercise involves the use of the 'Auto' data set, which we introduced in a previous lab session

# 13. Get the Auto data, redo the preprocessing
# 14. Apply E-M again with a family of your choice ("spherical", "diagonal", etc), letting BIC decide the best
#     number of clusters
# 15. Inspect and report the results of your clustering
#     Warning: do not directly plot() the results, it takes a long time
# 16. Use the clusplot() function in {cluster}
#     Like this:
#        library(cluster)
#        clusplot(Auto, z@bestResult@partition, color=TRUE, shade=TRUE, labels=2, lines=0)
#     please do consult ?clusplot.default


# Your code starts here ...
