import numpy as np
from timeit import timeit
import sys

cutoffs = [10, 100, 1000, 10000, 20000, 40000, 60000, 80000, 100000]
#cutoffs = [10, 100, 1000, 10000, 100000]
#cutoffs = [10, 100]

## check if images was loaded
if 'images' not in locals(): 
   raise ValueError('"load images to memory first')

# Let's double our images so that we have 100,000
images = np.concatenate((images, images), axis=0)
labels = np.concatenate((labels, labels), axis=0)

# Make sure you actually have 100,000 images and labels
print(np.shape(images))
print(np.shape(labels))


_store_many_funcs = dict(
    disk=store_many_disk, lmdb=store_many_lmdb, hdf5=store_many_hdf5
)

store_many_timings = {"disk": [], "lmdb": [], "hdf5": []}

for cutoff in cutoffs:
    print("Number of images: " + str(cutoff))
    for method in ("disk", "lmdb", "hdf5"):
        t = timeit(
            "_store_many_funcs[method](images_, labels_)",
            setup="images_=images[:cutoff]; labels_=labels[:cutoff]",
            number=1,
            globals=globals(),
        )
        store_many_timings[method].append(t)

        # Print out the method, cutoff, and elapsed time
        print(f"Method: {method}, Time usage: {t}")


