print("load modules")

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
import pickle
##############################################################

#sequential_read = True
sequential_read = False

##############################################################

######################### sqlite file - for lmdb and other, excpet hdf5
print("---reading sqlite for lmdb")
with sqlite3.connect("lmdb-jpg-10000.sqlite") as sql_conn:
    df = pd.read_sql_query("select * from meta;", sql_conn)
    all_keys = df['key'].tolist()

######################## hdf5 sqlite file
print("---reading sqlite for hdf5")
with sqlite3.connect("hdf5-jpg-10000.sqlite") as sql_conn:
    df_hdf5 = pd.read_sql_query("select * from meta;", sql_conn)
    all_keys_hdf5 = df_hdf5['key'].tolist()


######################## hdf5 sqlite file
print("---reading sqlite for hdf5")
with sqlite3.connect("hdf5-numpy-10000.sqlite") as sql_conn:
    df_hdf5_numpy = pd.read_sql_query("select * from meta;", sql_conn)
    all_keys_hdf5_numpy = df_hdf5_numpy['key'].tolist()

#############################################################
#############################################################
#timing_dict = {"lmdb": [], "hdf5": [], "scratch": [], "SLURM_TMPDIR": [], "SLURM_RAM_TMPDIR": [], "convert": [], "subset_of_paths": []}
#timing_dict = {"lmdb": [], "hdf5": [], "scratch": [], "convert": [], "subset_of_paths": []}
timing_dict = {"lmdb_convert": [], "lmdb_numpy": [], "hdf5_convert": [], "hdf5_numpy": [], "scratch_convert": [], "convert": [], "subset_of_paths": []}

#N_to_read = [100]
N_to_read = [100, 500, 1000, 3000, 5000, 8000, len(all_keys)]
timing_dict["N_to_read"] = N_to_read    


##############
## Do read  ##
##############

start_index = 0
max_index = len(all_keys)

#for N_of_files in [len(all_keys)]:    
for N_of_files in N_to_read:
    
    #########################
    ## Select keys to read ##
    #########################

    if sequential_read:    
        ## Sequentital read (next block of lines)
        key_list = all_keys[start_index:(start_index+N_of_files)]
        key_list_hdf5 = all_keys_hdf5[start_index:(start_index+N_of_files)]    
    else:
        ## Random access (next block of lines)
        random.seed(N_of_files)
        chosen_items = random.sample(range(start_index, max_index), N_of_files)
        key_list = [all_keys[k] for k in chosen_items]
        key_list_hdf5 = [all_keys_hdf5[k] for k in chosen_items]
        
    # monitor
    print("first five keys now: ")
    print(key_list[0:5])


    ##################
    ## LMDB, binary and convert         ##
    ##################
    print("read data from lmdb")
    env = lmdb.open("lmdb-jpg-10000.lmdb")
    
    im_ar = {}
    
    tic = time.time()
    convert_time = 0
    
    with env.begin() as lmdb_txn:
        print("read from lmdb")
        print(lmdb_txn.stat())
        for key in key_list:
            #print("workign with key: " + key)
            stored_image = lmdb_txn.get(key.encode())
            tic_convert = time.time()
            PIL_image = Image.open(io.BytesIO(stored_image))
            im_ar[key] =  np.asarray(PIL_image)
            toc_convert = time.time()
            convert_time = convert_time + (toc_convert-tic_convert)

    toc = time.time()
    env.close()

    timing_dict["lmdb_convert"].append(toc-tic)
    print("elapsed time: " + str(toc - tic))
    timing_dict["convert"].append(convert_time)
    
    
    ##################
    ## LMDB, numpy         ##
    ##################
    print("read data from lmdb")
    env = lmdb.open("lmdb-numpy-10000.lmdb")
    
    im_ar = {}
    
    tic = time.time()
    convert_time = 0
    
    with env.begin() as lmdb_txn:
        print("read from lmdb")
        print(lmdb_txn.stat())
        for key in key_list:
            stored_image = lmdb_txn.get(key.encode())
            im_ar[key] = pickle.loads(stored_image)

    toc = time.time()
    env.close()
    timing_dict["lmdb_numpy"].append(toc-tic)

    
    ####################
    ## HDF5, convert  ##
    ####################
    print('read data from hdf5')

    im_ar2 = {}
    tic = time.time()

    with h5py.File('hdf5-jpg-10000.hdf5', 'r') as f:
        dset = f['images']
        for key in key_list_hdf5:
            #print("workign with key: " + key)
            stored_image = dset[key]
            PIL_image = Image.open(io.BytesIO(stored_image))
            im_ar2[key] =  np.asarray(PIL_image)

    toc = time.time()
    timing_dict["hdf5_convert"].append(toc-tic)
    print("elapsed time: " + str(toc - tic))

    ##################
    ## HDF5, numpy  ##
    ##################
    print('read data from hdf5')

    im_ar2 = {}
    tic = time.time()


    with h5py.File('hdf5-numpy-10000.hdf5', 'r') as f:
        dset = f['images']
        for key in key_list_hdf5:
            #print("workign with key: " + key)
            
            ## get shape
            meta_data = df_hdf5_numpy[df_hdf5_numpy.key == key]
            shape0 = int(meta_data["shape0"])
            shape1 = int(meta_data["shape1"])
            shape2 = int(meta_data["shape2"])
            
            stored_image = dset[key]
            im_ar2[key] = stored_image.reshape((shape0, shape1, shape2))

    toc = time.time()
    timing_dict["hdf5_numpy"].append(toc-tic)
    print("elapsed time: " + str(toc - tic))

    
    ###################################################################
    ## Prepare to read from disk: Subset paths to read files from    ##
    ###################################################################
    tic1 = time.time()
    
    ## make a list of paths from data frame column
    paths_hdf5_all = df_hdf5["path"].tolist()

    ###### gets subset of paths
    if sequential_read:    
        ## Sequential
        paths_to_read = paths_hdf5_all[start_index:(start_index+N_of_files)]
    else:        
        ## random
        paths_to_read = [paths_hdf5_all[i] for i in key_list_hdf5 ]
    
    toc1 = time.time()
    timing_dict["subset_of_paths"].append(toc1-tic1)

    

    ##################
    ## scratch      ##
    ##################
    print("read data from scratch")
    ## add full paths
    paths_to_read_full = ["/scratch/ss13638/datasets/TextRec/data_10000/mnt/ramdisk/max/90kDICT32px/" + s for s in paths_to_read]
    im_ar3 = {}
    tic = time.time()
    #for key in key_list_hdf5:
    for ind in range(len(key_list_hdf5)):
        #print("workign with key: " + key)
        path_to_file = paths_to_read_full[ind]
        key = key_list_hdf5[ind]
        ## Read file directly to PIL_image
        ## PIL_image = Image.open(path_to_file)                       
        with open(path_to_file, mode='rb') as file: 
            stored_image = file.read() 
        PIL_image = Image.open(io.BytesIO(stored_image))
        im_ar3[key] =  np.asarray(PIL_image)
        
        
    toc = time.time()
    timing_dict["scratch_convert"].append(toc-tic)
    print("elapsed time: " + str(toc - tic))

print(timing_dict)



##################
## PLOT         ##
##################

N_array = timing_dict["N_to_read"]
del timing_dict["N_to_read"]
del timing_dict["convert"]
del timing_dict["subset_of_paths"]

df = pd.DataFrame(timing_dict, index=N_array)
df = df.drop(100)
df = df.round(3)

print(df)

lines = df.plot.line(style='.-', markersize = 20)
lines.set_xlabel("Number of images")
lines.set_ylabel("Time for reading (s)")


if sequential_read:    
    plt.savefig('./read_numpy_sequential.png')
else:
    plt.savefig('./read_numpy_random.png')

