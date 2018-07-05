rm(list=ls(all=TRUE))
#Set the directory 
setwd("D:\\My Files\\Automobile")

#Import the data and read the file.
traindata<-read.csv(file = "automobiledata.csv", header = TRUE,na.strings=c("","?"))

#Checking structure of traindata.
str(traindata)
#Checking summary of traindata.
summary(traindata)

#Checking the NA values per variable in train and test dataset.
na_train <-sapply(traindata, function(x) sum(is.na(x)))
na_train <- data.frame(na_train)

#Checking total NA values present in test and train dataset.
sum(is.na(traindata))

#Checking names of variables present in train dataset.
names(traindata)

#Checking if data is having more than 20% NA's.
library(DMwR)
colMeans(is.na(traindata)) > 0.2

#Boxplot for normalized.losses variable.
boxplot(traindata$normalized.losses)

#Plotting histogram for price.
hist(traindata$price,border ="black",col="blue",main = "Historgram Of Price", xlab = "Price Of Cars", ylab = "Frequency Of Cars")

#Plotting
library(ggplot2)
ggplot(traindata, aes(make,fill=aspiration)) + 
  geom_bar(width=0.8) +
  facet_wrap(~ aspiration) +
  labs(y = 'Frequency of Cars', 
       x = 'Make') +
  ggtitle("Standard vs Turbo") + 
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(axis.text.x = element_text(face="bold", angle=45))

#Applying Knn Imputation
library(VIM)
traindata2 <- kNN(traindata,variable = c("normalized.losses","num.of.doors","bore","stroke",
                                         "horsepower","peak.rpm","price"),k=3)

#Checking Summary.
summary(traindata2)

#Removing the Extra columns
traindata2 <- subset(traindata2, select=symboling:price)
str(traindata2)

#Checking NA's.
sum(is.na(traindata2))

#Plotting
library(ggplot2)
ggplot(traindata2, aes(x=make, y=symboling)) + geom_point()

#Scatter Plot
library("car")
scatterplot(price ~ city.mpg, data = traindata2,main="Price and Fuel",xlab = 'City(in mpg)',ylab = 'Price of Car')

#Plotting
library(ggplot2)
ggplot(traindata2, aes(make,fill=aspiration)) + 
  geom_bar(width=0.8) +
  facet_wrap(~ aspiration) +
  labs(y = 'Frequency of Cars', 
       x = 'Make') +
  ggtitle("Standard vs Turbo") + 
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(axis.text.x = element_text(face="bold", angle=45))


#Plotting
library(ggplot2)
ggplot(traindata2, aes(make,fill=fuel.type)) + 
  geom_bar(width=0.8) +
  facet_wrap(~fuel.type) +
  labs(y = 'Frequency of Cars', 
       x = 'Make') +
  ggtitle("diesel vs gas") + 
  theme(plot.title = element_text(hjust = 0.5)) +
  theme(axis.text.x = element_text(face="bold", angle=45))


#Feature Selection Process.
library(Boruta)

set.seed(123)

boruta.train <- Boruta(symboling~.,data = traindata2, doTrace = 2)
print(boruta.train)


plot(boruta.train, xlab = "", xaxt = "n")

lz<-lapply(1:ncol(boruta.train$ImpHistory),function(i)
boruta.train$ImpHistory[is.finite(boruta.train$ImpHistory[,i]),i])
names(lz) <- colnames(boruta.train$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels),
at = 1:ncol(boruta.train$ImpHistory), cex.axis = 0.7)


#To know the selected attributes.
getSelectedAttributes(boruta.train)

#Build a dataframe to check confirmed and rejected features.
boruta.df <- attStats(boruta.train)
print(boruta.df)


#Split the data into train and test data sets
rows=seq(1,nrow(traindata2),1)
trainRows=sample(rows,(70*nrow(traindata2))/100)
train = traindata2[trainRows,] 
test = traindata2[-trainRows,]


#Correlation Checking.
cor(traindata2$price, traindata2$horsepower)
#cor(traindata2$price, traindata2$city.mpg)
cor(traindata2$price,traindata2$symboling)
cor(traindata2$symboling, traindata2$normalized.losses)
cor(traindata2$price, traindata2$engine.size)
cor(traindata2$price, traindata2$wheel.base)
cor(traindata2$price, traindata2$peak.rpm)
cor(traindata2$engine.size, traindata2$horsepower)

#Build Multiple Linear Regression Model
mlnfit <- lm(price~ horsepower + wheel.base + normalized.losses, data=train)
summary(mlnfit)

# Evaluate Collinearity
vif(mlnfit) # variance inflation factors 


# Choose a VIF cutoff[recommended '2']
threshold=2
# Create function to sequentially drop the variable with the largest VIF until 
# all variables have VIF > cutoff
library(plyr)
flag=TRUE
viftable=data.frame()
while(flag==TRUE) {
  vfit=vif(mlnfit)
  viftable=rbind.fill(viftable,as.data.frame(t(vfit)))
  if(max(vfit)>threshold) { mlnfit=
    update(mlnfit,as.formula(paste(".","~",".","-",names(which.max(vfit))))) }
  else { flag=FALSE } }
# Look at the final model
print(mlnfit)
# And associated VIFs
print(vfit)
# And show the order in which variables were dropped
print(viftable)


#Setting Confidence Interval
confint(mlnfit,conf.level=0.95)

#Using par as 1,1 for viewing one plot at a time.
par(mfrow=c(2,2))

#Plotting model diagnostics.
plot(mlnfit)

#crossvalidation

library(glmnet)
library(caret)
set.seed(123)

glmnet_grid <- expand.grid(alpha = c(0,  .1,  .2, .4, .6, .8, 1),
                           lambda = seq(.01, .2, length = 20))
glmnet_ctrl <- trainControl(method = "cv", number = 10)
glmnet_fit <- train(price ~ horsepower + wheel.base + normalized.losses,data =train,
                    method = "glmnet",
                    preProcess = c("center", "scale"),
                    tuneGrid = glmnet_grid,
                    trControl = glmnet_ctrl)

glmnet_fit #R squared higher the better

#test dataset prediction

pricePred <- predict(mlnfit,test)
test$price <- pricePred


actuals_preds <- data.frame(cbind(actuals=test$price, predicteds=pricePred))  # make actuals_predicteds dataframe.
correlation_accuracy <- cor(actuals_preds)
head(actuals_preds)


#MAPE lower the better
#mix_max accuracy higher the better.
min_max_accuracy <- mean(apply(actuals_preds, 1, min) / apply(actuals_preds, 1, max))  
mape <- mean(abs((actuals_preds$predicteds - actuals_preds$actuals))/actuals_preds$actuals)  



#train dataset prediction.
pricePred1 <- predict(mlnfit,train)


actuals_preds1 <- data.frame(cbind(actuals=train$price, predicteds=pricePred1))  # make actuals_predicteds dataframe.
str(actuals_preds1)
correlation_accuracy <- cor(actuals_preds1)
head(actuals_preds1)


#MAPE lower the better
min_max_accuracy1 <- mean(apply(actuals_preds1, 1, min) / apply(actuals_preds1, 1, max))  
mape1 <- mean(abs((actuals_preds1$predicteds - actuals_preds1$actuals))/actuals_preds1$actuals)  


#write.csv(final_result, file = "output1.csv", row.names = FALSE)