## Code based on https://keras.rstudio.com/  and https://github.com/ageron/handson-ml2/blob/master/10_neural_nets_with_keras.ipynb

#packrat::restore()

####################
## Specify Python Environment
####################
library(reticulate)
## CPU version
use_virtualenv("./venv", required = TRUE)
## GPU version
#use_virtualenv("./venv_gpu", required = TRUE)


####################
## Check tensorflow and keras versions
####################
library(tensorflow)
print("Tensorflow version")
tensorflow::tf_version()
print("Keras version")
packageVersion("keras")
print("Num GPUs Available:")
## Look at configuration (are there GPUs available?)
tf_config()
num_gpu = tensorflow::tf_gpu_configured()


####################
## Get data
####################
library(keras)
mnist = dataset_mnist()
X_train_full <- mnist$train$x
Y_train_full <- mnist$train$y
X_test <- mnist$test$x
Y_test <- mnist$test$y

X_train_full = X_train_full / 255
X_test = X_test / 255

print("X_train_full.shape")
X_dims <- dim(X_train_full)

X_train = X_train_full[1:5000, 1:X_dims[2], 1:X_dims[3]]
X_valid = X_train_full[5001:X_dims[1], 1:X_dims[2], 1:X_dims[3]]
Y_train = Y_train_full[1:5000]
Y_valid = Y_train_full[5001:X_dims[1]]

class_names <- c("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")

Y_train <- to_categorical(Y_train, 10)
Y_valid <- to_categorical(Y_valid, 10)
Y_test <- to_categorical(Y_test, 10)

print("X_valid.shape")
dim(X_valid)
print("X_test.shape")
dim(X_test)


####################
## Define model
####################
model <- keras_model_sequential() 
model %>% 
  layer_flatten(input_shape = c(28, 28)) %>%
  layer_dense(units = 300, activation = 'relu') %>% 
  layer_dense(units = 100, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

summary(model)

k_clear_session()

## R seed
set.seed(42)
## Tensoflow seed, if needed
## https://tensorflow.rstudio.com/reference/tensorflow/use_session_with_seed/
#tensorflow::use_session_with_seed(seed = 42, disable_gpu = TRUE, disable_parallel_cpu = TRUE)

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_sgd(),
  metrics = c('accuracy')
)

####################
## Train model
####################
history <- model %>% fit(
  X_train, Y_train, 
  epochs = 10, #epochs = 30,
  validation_data = list(X_valid, Y_valid)
)

####################
## Make and Save learning curves plot
####################
library("ggplot2")
pdf("learning_curve.pdf")
plot(history)
dev.off()

####################
## Evaluate
####################
print("Evaluate on test")
model
model$evaluate(X_test, Y_test)
