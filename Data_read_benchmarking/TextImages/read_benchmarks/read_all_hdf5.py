## srun --ntasks-per-node=1 --nodes 1 --mem=60GB -t1:00:00 --pty /bin/bash

import numpy as np
from PIL import Image
import lmdb
from random import sample
import sqlite3
import pandas as pd
import io
import time
import os
import random
import h5py
import matplotlib.pyplot as plt
import sys

##############################################################

#print("---reading sqlite for hdf5")
######################## hdf5 sqlite file
#with sqlite3.connect("/scratch/work/public/datasets/TextRecognitionData_VGG_Oxford/hdf5/TextImages-hdf5.sqlite") as sql_conn:
#  df_hdf5 = pd.read_sql_query("select * from meta;", sql_conn)
#all_keys_hdf5 = df_hdf5['key'].tolist()
all_keys_hdf5 = list(range(8919273))

#############################################################
#############################################################
timing_dict = {"hdf5": []}
N_to_read = len(all_keys_hdf5)
timing_dict["N_to_read"] = N_to_read    

##############################################################
print('read data from hdf5')

im_ar2 = {}
tic = time.time()

with h5py.File('/scratch/work/public/datasets/TextRecognitionData_VGG_Oxford/hdf5/TextImages.hdf5', 'r') as f:
  dset = f['images']
  for key in all_keys_hdf5:
    if key % 10000 == 0: 
      print("workign with key: " + str(key))
    stored_image = dset[key]
    #im_ar2[key] = stored_image
    #print(data)
    PIL_image = Image.open(io.BytesIO(stored_image))
    np_array =  np.asarray(PIL_image)
    #im_ar2[key] = dset[key]


toc = time.time()
timing_dict["hdf5"].append(toc-tic)
print("elapsed time: " + str(toc - tic))

print("size of array: " + str(sys.getsizeof(im_ar2)))
print(timing_dict)

