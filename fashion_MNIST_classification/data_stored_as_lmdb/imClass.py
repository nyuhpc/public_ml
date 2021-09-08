####################
## Code based on https://github.com/ageron/handson-ml2/blob/master/10_neural_nets_with_keras.ipynb
## Some ideas from https://www.pyimagesearch.com/2018/12/24/how-to-use-keras-fit-and-fit_generator-a-hands-on-tutorial/
####################

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt

####################
## Verify version
####################
print("tf.__version__")
print(tf.__version__)
print("keras.__version__")
print(keras.__version__)
#print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))

#############################################################################
import lmdb
import os
import shutil
import pickle

from gen_func import *

##########
# Prepare data access
##########
env_train = lmdb.open("x-numpy-train.lmdb")
env_valid = lmdb.open("x-numpy-valid.lmdb")
env_test = lmdb.open("x-numpy-valid.lmdb")

y_train = pickle.load( open( "y_train.pkl", "rb" ) )
y_valid = pickle.load( open( "y_valid.pkl", "rb" ) )
y_test = pickle.load( open( "y_test.pkl", "rb" ) )

####################
## Define model
####################
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.layers
model.summary()

model.compile(loss="categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

####################
## Train model
####################
batch_size = 32
EPOCHS = 10
history = model.fit_generator(
    generator = lmdb_data_generator(
        env_x = env_train, y_labels = y_train, batch_size = batch_size),
    validation_data = lmdb_data_generator(
        env_x = env_valid, y_labels = y_valid, batch_size = batch_size),
    
    #steps_per_epoch = 10,
    #validation_steps = 5,
    
    steps_per_epoch = env_train.stat()["entries"] // batch_size,
    validation_steps = env_valid.stat()["entries"] // batch_size,
    
    epochs=EPOCHS)

print("history.params")
print(history.params)

print("history.epoch")
print(history.epoch)

print("history.history.keys()")
print(history.history.keys())

####################
## Make and Save learning curves plot
####################
import pandas as pd

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.savefig("keras_learning_curves_plot.png", dpi=300)

####################
## Evaluate on Test
####################
print(model)
print(model.evaluate(X_test, y_test))

## on validation if needed
# model.evaluate(
#     lmdb_data_generator(
#         env_x = env_valid, y_labels = y_valid, batch_size = batch_size), 
#         steps = env_valid.stat()["entries"] // batch_size)

# on test
model.evaluate(
    lmdb_data_generator(
        env_x = env_test, y_labels = y_test, batch_size = batch_size), 
        steps = env_test.stat()["entries"] // batch_size)

####################
## Cleaning up
####################
env_train.close()
env_valid.close()
env_test.close()

