####################################################################
# Machine Learning - MIRI
# Lluís A. Belanche

# LAB 1: Data pre-processing (practise)
# version of February 2019
####################################################################

# This exercise involves the use of the 'Auto' data set, which can be found in the file 'Auto.data'. 
# The file contains a number of variables for cars.

graphics.off()      # reset/close all graphical devices 


##### Reading the file 'Auto.data' (data on cars)

Auto <- read.table("Auto.data", header=TRUE, na.strings="?")

# put proper country of origin
Auto[,"origin"] <- factor(c("USA","EU","Japan")[Auto[,"origin"]])

# convert "miles per gallon" to "liters per km"
Auto[,"mpg"] <- 235.4/Auto[,"mpg"]
colnames(Auto)[which(colnames(Auto)=="mpg")] <- "l.100km"

# The car name is not useful for modelling, but it may be handy to keep it as the row name

# WARNING! surprisingly, car names are not unique, so we first prefix them by their row number

Auto$name <- paste (1:nrow(Auto), Auto$name)
rownames(Auto) <- Auto$name
Auto <- subset (Auto, select=-name)

# Now we go for the cylinders
table(Auto$cylinders)

# that's strange, some cars have an odd number of cylinders (are these errors?)

subset(Auto,cylinders==3)

# These Mazdas wear a Wankel engine, so this is correct

subset(Auto,cylinders==5)

# Yes, these Audis displayed five-cylinder engines, so the data is correct

# but, from summary(Auto) above we see that horsepower has 5 NA's that we'll need to take care of later ...

# so this is your departing data set
summary(Auto)
#attach(Auto)

# maybe you remember that plot from Lecture 1 ...
with (Auto, Auto.lm <<- lm(l.100km ~ horsepower, Auto))

plot(Auto[,"horsepower"],Auto[,"l.100km"],
     pch=20,
     xlab="horsepower",ylab="fuel consumption (l/100km)",
     main="Linear regression")

# add regression line
a <- Auto.lm$coefficients["(Intercept)"]
b <- Auto.lm$coefficients["horsepower"]
abline(a=a,b=b,col="blue")
text(50,25,sprintf("y(x)=%.3fx+%.2f",b,a),col="red",pos=4)

# In order to crate quick LaTeX code, try this:
  
install.packages("xtable")
library(xtable)

xtable(Auto[1:4,])
xtable(Auto.lm)

# Was that nice? 
# this is a list of R objects that can be embedded into a LaTeX table code:

methods(xtable)

###################################################################################
# Exercise for the lab session
###################################################################################

# 1. print the dimensions of the data set 

# 2. identify possible target variables according to classification or regression problems

# 3. inspect the first 4 examples and the predictive variables 6 and 7 for the tenth example

# 4. perform a basic inspection of the dataset. Have a look at the minimum and maximum values for each variable; find possible errors and abnormal values (outliers); find possible missing values; decide which variables are really continuous and which are really categorical and convert them

# 5. make a decision on a sensible treatment for the missing values and apply it; 

#    WARNING: 'origin' is categorical and cannot be used for knn imputation, unless you make it binary temporarily

# 6. derive one new continuous variable: weight/horsepower; derive one new categorical variable: sports_car, satisfying horsepower > 1.2*mean(horsepower) AND acceleration < median(acceleration); do you think this new variable is helpful in predicting 'origin' ?

# 7. create a new dataframe that gathers everything and inspect it again

# 8. perform a graphical summary of some of the variables (both categorical and continuous)

# 9. perform a graphical comparison between some pairs of variables (both categorical and continuous)

# 10. do any of the continuous variables "look" Gaussian? can you transform some variable so that it looks more so?

# 11. create a new dataframe that gathers everything and inspect it again; consider 'origin' as the target variable; perform a basic statistical analysis as indicated in SECTION 9

# 12. shuffle the final dataset and save it into a file for future use

# Your code starts here ...

