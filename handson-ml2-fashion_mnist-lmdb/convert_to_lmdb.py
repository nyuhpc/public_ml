import tensorflow as tf
from tensorflow import keras
import numpy as np
import lmdb
import os
import shutil
import pickle

##########
## Verify version
##########
print("tf.__version__")
print(tf.__version__)
print("keras.__version__")
print(keras.__version__)
#print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))

##########
## Get data
##########
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
print("X_test.shape")
print(X_test.shape)

####################
## Save data to LMDB file
####################

##########
# Training data to lmdb
##########
filename = "x-numpy-train.lmdb"
if os.path.exists(filename): shutil.rmtree(filename)
env_train = lmdb.open(filename, map_size=int(1e9))

for file_index in range(y_train.shape[0]):
    with env_train.begin(write=True) as lmdb_txn:
        value = X_train[file_index]
        key = f"{file_index:07}"
        lmdb_txn.put(key.encode(), pickle.dumps(value))  

env_train.close()

with open("y_train.pkl", 'wb') as f:
    pickle.dump(y_train, f)

##########
# Validation data to lmdb
##########
filename = "x-numpy-valid.lmdb"
if os.path.exists(filename): shutil.rmtree(filename)
env_valid = lmdb.open(filename, map_size=int(1e9))

for file_index in range(y_valid.shape[0]):
    with env_valid.begin(write=True) as lmdb_txn:
        value = X_valid[file_index]
        key = f"{file_index:07}"
        lmdb_txn.put(key.encode(), pickle.dumps(value))  

env_valid.close()

with open("y_valid.pkl", 'wb') as f:
    pickle.dump(y_valid, f)

##########
# Test data to lmdb
##########
filename = "x-numpy-test.lmdb"
if os.path.exists(filename): shutil.rmtree(filename)
env_test = lmdb.open(filename, map_size=int(1e9))

for file_index in range(y_test.shape[0]):
    with env_test.begin(write=True) as lmdb_txn:
        value = X_test[file_index]
        key = f"{file_index:07}"
        lmdb_txn.put(key.encode(), pickle.dumps(value))  

env_test.close()

with open("y_test.pkl", 'wb') as f:
    pickle.dump(y_test, f)
