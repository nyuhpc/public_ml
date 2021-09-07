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
##############################################################

#sequential_read = True
sequential_read = False

convert_to_numpy = False
#convert_to_numpy = True

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

#############################################################
#############################################################
timing_dict = {"lmdb": [], "hdf5": [], "scratch": [], "SLURM_TMPDIR": [], \
               "SLURM_RAM_TMPDIR": [], "convert": [], "subset_of_paths": [],\
               "lmdb_local": [], "hdf5_local": []}
#timing_dict = {"lmdb": [], "hdf5": [], "scratch": [], "convert": [], "subset_of_paths": []}

#N_to_read = [100]
N_to_read = [100, 500, 1000, 3000, 5000, 8000, len(all_keys)]
timing_dict["N_to_read"] = N_to_read    

###############
## Extract data ##
###############
print("copy files")
tic_1 = time.time()
#os.system("cp -r data_10000/* $SLURM_TMPDIR/")
os.system("tar -C $SLURM_TMPDIR -xf data_10000.tar.gz")
slurm_tmpdir_copy_time = time.time() - tic_1
print("copy to SLURM_TMPDIR process took (included in total bellow): " + str(slurm_tmpdir_copy_time))

tic_1 = time.time()
os.system("cp -r lmdb-jpg-10000.lmdb $SLURM_TMPDIR/")
print("copy lmdb file time (NOT included in total bellow): " + str(time.time() - tic_1))

tic_1 = time.time()
os.system("cp -r hdf5-jpg-10000.hdf5 $SLURM_TMPDIR/")
print("copy hdf5 file time (NOT included in total bellow): " + str(time.time() - tic_1))



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
        key_list.sort()
        key_list_hdf5 = [all_keys_hdf5[k] for k in chosen_items]
        key_list_hdf5.sort()
        
    # monitor
    print("first five keys now: ")
    print(key_list[0:5])


    ##################
    ## LMDB         ##
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
            #print("value read")
            #print(stored_image)
            if convert_to_numpy:
                tic_convert = time.time()
                PIL_image = Image.open(io.BytesIO(stored_image))
                im_ar[key] =  np.asarray(PIL_image)
                toc_convert = time.time()
                convert_time = convert_time + (toc_convert-tic_convert)
            else:
                im_ar[key] =  stored_image
    toc = time.time()
    env.close()

    timing_dict["lmdb"].append(toc-tic)
    print("elapsed time: " + str(toc - tic))
    timing_dict["convert"].append(convert_time)
    
    
    ##################
    ## LMDB, slurm ##
    ##################
    print("read data from lmdb-local")
    
    env = lmdb.open(os.getenv("SLURM_TMPDIR") + "/lmdb-jpg-10000.lmdb", 
                    readonly=True, lock=False)
    
    im_ar_l = {}
    
    tic = time.time()
    convert_time = 0
    
    with env.begin() as lmdb_txn:
        print("read from lmdb")
        print(lmdb_txn.stat())
        for key in key_list:
            stored_image = lmdb_txn.get(key.encode())
            if convert_to_numpy:
                PIL_image = Image.open(io.BytesIO(stored_image))
                im_ar_l[key] =  np.asarray(PIL_image)
            else:
                im_ar_l[key] =  stored_image
    toc = time.time()
    env.close()

    timing_dict["lmdb_local"].append(toc-tic)
    print("elapsed time: " + str(toc - tic))
    
    
    ##################
    ## HDF5         ##
    ##################
    print('read data from hdf5')
    
    im_ar2 = {}
    tic = time.time()
    
    with h5py.File('hdf5-jpg-10000.hdf5', 'r') as f:
        dset = f['images']
        if convert_to_numpy:
            chosen_stored_images = dset[key_list_hdf5]
            im_ar2 = [np.asarray(Image.open(io.BytesIO(stored_image))) for stored_image in chosen_stored_images]
        else:
            chosen_stored_images = dset[key_list_hdf5]
    
    toc = time.time()
    timing_dict["hdf5"].append(toc-tic)
    print("elapsed time: " + str(toc - tic))

    ##################
    ## HDF5-local   ##
    ##################
    print('read data from hdf5-local')
    
    im_ar2_l = {}
    tic = time.time()
    
    with h5py.File(os.getenv("SLURM_TMPDIR") + "/hdf5-jpg-10000.hdf5", 'r') as f:
        dset = f['images']
        if convert_to_numpy:
            chosen_stored_images = dset[key_list_hdf5]
            im_ar2_l = [np.asarray(Image.open(io.BytesIO(stored_image))) for stored_image in chosen_stored_images]
        else:
            im_ar2_l = dset[key_list_hdf5]
            
    toc = time.time()
    timing_dict["hdf5_local"].append(toc-tic)
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
        if convert_to_numpy:
            PIL_image = Image.open(io.BytesIO(stored_image))
            im_ar3[key] =  np.asarray(PIL_image)
        else:
            im_ar3[key] =  stored_image
        
        
    toc = time.time()
    timing_dict["scratch"].append(toc-tic)
    print("elapsed time: " + str(toc - tic))

    
    
    ##################
    ## SLURM_TMPDIR ##
    ##################
    print("read data from SLURM_TMPDIR")
    ## add full paths
    paths_to_read_full = [os.environ['SLURM_TMPDIR']  + "/data_10000/mnt/ramdisk/max/90kDICT32px/" + s for s in paths_to_read]
    im_ar4 = {}
    tic = time.time()

    for ind in range(len(key_list_hdf5)):
        path_to_file = paths_to_read_full[ind]
        #print("workign with key: " + str(key) + " and path " + path_to_file)
        key = key_list_hdf5[ind]
        ## Read file directly to PIL_image
        ## PIL_image = Image.open(path_to_file)
        with open(path_to_file, mode='rb') as file:
            stored_image = file.read()
        if convert_to_numpy:
            PIL_image = Image.open(io.BytesIO(stored_image))
            im_ar4[key] =  np.asarray(PIL_image)
        else:
            im_ar4[key] =  stored_image
    toc = time.time()
    timing_dict["SLURM_TMPDIR"].append(toc-tic)
    print("elapsed time: " + str(toc - tic))


    timing_dict["SLURM_RAM_TMPDIR"].append(0)
    # ######################
    # ## SLURM_RAM_TMPDIR ##
    # ######################
    # print('read data from SLURM_RAM_TMPDIR')
    # ## add full paths
    # paths_to_read_full = [os.environ['SLURM_RAM_TMPDIR'] + "/" + "mnt/ramdisk/max/90kDICT32px/" + s for s in paths_to_read]
    # im_ar5 = {}
    # tic = time.time()
    # 
    # #for key in key_list_hdf5:
    # for ind in range(len(key_list_hdf5)):
    #     #print("workign with key: " + key)
    #     path_to_file = paths_to_read_full[ind]
    #     key = key_list_hdf5[ind]
    #     ## Read file directly to PIL_image
    #     ## PIL_image = Image.open(path_to_file)
    #     with open(path_to_file, mode='rb') as file:
    #         stored_image = file.read()
    #     if convert_to_numpy:
    #         PIL_image = Image.open(io.BytesIO(stored_image))
    #         im_ar5[key] =  np.asarray(PIL_image)
    #     else:
    #         im_ar5[key] =  stored_image
    # toc = time.time()
    # timing_dict["SLURM_RAM_TMPDIR"].append(toc-tic)
    # print("elapsed time: " + str(toc - tic))

    ###############################
    ## update staring read index ##
    ###############################
    # start_index = start_index + N_of_files

print(timing_dict)


##################################
## Adjust for read time         ##
##################################

#timing_dict["SLURM_TMPDIR"] = \
#  [slurm_tmpdir_copy_time + v for v in timing_dict["SLURM_TMPDIR"]] 
#timing_dict["SLURM_RAM_TMPDIR"] = \
#  [slurm_tmpRAMdir_copy_time + v for v in timing_dict["SLURM_RAM_TMPDIR"]] 


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
#print(df.apply(lambda x: x.astype("str").str.pad(10, side ='left')))

lines = df.plot.line(style='.-', markersize = 20)
lines.set_xlabel("Number of images")
lines.set_ylabel("Time for reading (s)")

lines.set_title("NOTE: time reported for SLURM dirs\n" +
                "does not take into account time \n" + 
                "taken to copy data to those dirs")

if sequential_read and convert_to_numpy:    
    plt.savefig('./read_binary_sequential_convert.png')
elif sequential_read and not convert_to_numpy:
    plt.savefig('./read_binary_sequential.png')
elif not sequential_read and convert_to_numpy:
    plt.savefig('./read_binary_random_convert.png')
elif not sequential_read and not convert_to_numpy:    
    plt.savefig('./read_binary_random.png')

######################################################################
# Compare read-convert each, vs read - then convert the whole list
if not convert_to_numpy: 
    def convert(key):
        try:
           np.asarray( Image.open(io.BytesIO(im_ar3[key])) )
        except:
           print("error with reading key " + str(key))
           None
    
    tic = time.time()
    res = [convert(key) for key in key_list_hdf5 ]
    toc = time.time()
    print("Time to convert the whole array is " + str(toc - tic))
## result:
## convert sum from one by one is 1.84 s
## Time to convert the whole array is 1.83

