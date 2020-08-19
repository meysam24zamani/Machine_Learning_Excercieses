####################################################################
# Machine Learning - MIRI Master
# Lluís A. Belanche

# LAB 10: Radial Basis Function Network (Part 2)
# version of April 2019
####################################################################


####################################################################
## Exercise

## We continue with Example 1: regression of a 1D function

## We are interested in studying the influence of sample size on the fit.
## The idea is that you embed the code in Part 1 into a couple of handy functions and leave
## the learning sample size (N) as a parameter.

## These are the learning sample sizes you are going to study

Ns <- c(25,50,100,200,500)

## You are asked to report the chosen lambda and the final test error (on the same test set),
## plot the learned function against the data and the true genearting function and see if the fit
## is better/worse/equal as a function of N and in what sense it is better/worse/equal

# Your code starts here ...
