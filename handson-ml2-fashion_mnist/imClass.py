## Code based on https://github.com/ageron/handson-ml2/blob/master/10_neural_nets_with_keras.ipynb
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt


## Verify version
print("tf.__version__ ")
print(tf.__version__)
print("keras.__version__ ")
print(keras.__version__)

## Get data
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

print("X_train_full.shape")
print(X_train_full.shape)

X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

print("X_valid.shape")
print(X_valid.shape)
print("X_test.shape ")
print(X_test.shape)


## Define model
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

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

## Train model
history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))

print("history.params")
print(history.params)

print("history.epoch")
print(history.epoch)

print("history.history.keys()")
print(history.history.keys())

## Make and Save learning curves plot
import pandas as pd

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.savefig("keras_learning_curves_plot.png", dpi=300)


print(model)
print(model.evaluate(X_test, y_test))

