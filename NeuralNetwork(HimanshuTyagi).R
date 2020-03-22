# Libraries to be installed and loaded
library(keras)
library(mlbench) 
library(dplyr)
library(magrittr)
library(neuralnet)

data("BostonHousing") #Data of your choosing
data <- BostonHousing
str(data)

data %<>% mutate_if(is.factor, as.numeric) #converting all non numeric data to numeric

#mdev is dependent and others are independent
n <- neuralnet(medv ~ crim+zn+indus+chas+nox+rm+age+dis+rad+tax+ptratio+b+lstat,data = data,hidden = c(10,5),linear.output = F,lifesign = 'full',rep=1) #hidden telss the specification of the neural network 
plot(n,col.hidden = 'darkgreen',col.hidden.synapse = 'darkgreen',show.weights = T,information = T,fill = 'lightblue') #visualize your neural network

# Matrix creation
data <- as.matrix(data)
dimnames(data) <- NULL

# creating a partition of my data into test and training
set.seed(222)
ind <- sample(2, nrow(data), replace = T, prob = c(.85, .15)) #prob tells the distribution being done of the testing and training data
training <- data[ind==1,1:13] #13 independent variables
test <- data[ind==2, 1:13]
trainingtarget <- data[ind==1, 14] #taking the last value which is a dependent variable
testtarget <- data[ind==2, 14]

# Normalize
m <- colMeans(training)
s <- apply(training, 2, sd) #sd is standard deviation
training <- scale(training, center = m, scale = s) #using scale you normalize the data
test <- scale(test, center = m, scale = s)

# customizing my keras model
model <- keras_model_sequential()
model %>% 
         layer_dense(units = 10, activation = 'relu', input_shape = c(13)) %>% #1st layer
         layer_dense(units = 5, activation = 'relu', input_shape = c(13)) %>% #2nd layer
         layer_dense(units = 1) #last layer(output)

# Compile the model
model %>% compile(loss = 'mse',optimizer = 'rmsprop',metrics = 'mae')

# Fit Model
mymodel <- model %>%
         fit(training,trainingtarget,epochs = 250,batch_size = 16,validation_split = 0.2)
# Evaluate
model %>% evaluate(test, testtarget)
pred <- model %>% predict(test)
mean((testtarget-pred)^2)
plot(testtarget, pred)