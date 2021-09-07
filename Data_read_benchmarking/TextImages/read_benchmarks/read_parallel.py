#srun --ntasks-per-node=16 --nodes 1 --mem=20GB -t1:00:00 --pty /bin/bash

from joblib import Parallel, delayed, parallel_backend
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
import math


################################### 
## Correct affinity              ##
###################################

## see current affinity 
import os; os.sched_getaffinity(0)
# reset affinity - read about affinity in man taskset
os.system("taskset -p 0xFFFFFFFF %d" % os.getpid())
# check
os.sched_getaffinity(0)

###################
## read metadata ##
###################

print("---reading sqlite for hdf5")
######################## hdf5 sqlite file
with sqlite3.connect("/scratch/work/public/datasets/TextRecognitionData_VGG_Oxford/hdf5/TextImages-hdf5.sqlite") as sql_conn:
    df_hdf5 = pd.read_sql_query("select * from meta;", sql_conn)
    all_keys_hdf5 = df_hdf5['key'].tolist()


################################### 
## copy files to SLURM_TMPDIR    ##
###################################
print("copy data to SLURM_TMPDIR")

print("---extracting files to SLURM_TMPDIR")
######################## extract files to SLURM_TMPDIR
tic_1 = time.time()
os.system("tar -C $SLURM_TMPDIR -xf mjsynth.tar.gz")
print("extracting process took (NOT included in total bellow): " + str(time.time() - tic_1))

tic_1 = time.time()
os.system("cp -r /scratch/work/public/datasets/TextRecognitionData_VGG_Oxford/lmdb/TextImages.lmdb $SLURM_TMPDIR/")
print("copy lmdb file time (NOT included in total bellow): " + str(time.time() - tic_1))

tic_1 = time.time()
os.system("cp -r /scratch/work/public/datasets/TextRecognitionData_VGG_Oxford/hdf5/TextImages.hdf5 $SLURM_TMPDIR/")
print("copy hdf5 file time (NOT included in total bellow): " + str(time.time() - tic_1))


#############################################################
#############################################################
timing_dict = {"lmdb": [], "hdf5": [], "lmdb_local": [], "hdf5_local": [], "SLURM_TMPDIR": [], "subset_of_paths": []}

## Assuming we have 16 cpus avaialble on node
## Do 16 codes two times - if any caching is happeing it will happen for first 16 cores run, and the rest of benchmark will happen on already cached data
cpus_options = [16, 16, 8, 4, 2, 1]
#cpus_options = [2, 1]
timing_dict["cpus_options"] = cpus_options

## Read fixed number of images for all experiments
N_to_read = int(1e5)

##############
## Do read  ##
##############

## Don't start from the same index all the time - Lustre will cache files and timing will be off
start_index = 0

####################
##  LMDB function ##
####################

## we open env in every process. Opening it for every key is slow (made reading twice as slow) - thus pass chunk of keys 
def read_fig_lmdb(key_sublist):
  ret_ar = []
  #print("process id:" + str(os.getpid()))
  #print("work with file " + lmdb_path)
  with lmdb.open(lmdb_path, readonly=True, lock=False) as env:
    with env.begin() as lmdb_txn:
      for key in key_sublist:
        #print(key)
        stored_image = lmdb_txn.get(key.encode())
        ret_ar.append(stored_image)
  return ret_ar

####################
##  LMDB scratch ##
####################

print("read lmdb scratch")

lmdb_path = "/scratch/work/public/datasets/TextRecognitionData_VGG_Oxford/lmdb/TextImages.lmdb"

for cpus_number in cpus_options:
    print("-----start index:" + str(start_index))
    ## generate updated key_list
    key_list = [f"{i:07}" for i in list(range(start_index+1, start_index+N_to_read+1))]
    
    ## split key_list to chunks
    n = math.ceil(len(key_list)/cpus_number)
    key_list_chunks = [key_list[i:i + n] for i in range(0, len(key_list), n)]     
    
    tic = time.time()
    res = Parallel(n_jobs=cpus_number, prefer = "processes", verbose=5)(delayed(read_fig_lmdb)(key_list) \
                                                                        for key_list in key_list_chunks )
    toc = time.time()
    timing_dict["lmdb"].append(toc-tic)
    # start_index = start_index + N_to_read

## flatten res 
tic_f = time.time()
res = [val for sublist in res for val in sublist]
print("flatten time for lmdb:" + str(time.time()-tic_f))
print("check loaded data")
print(res[1])

####################
##  LMDB local    ##
####################

print("read lmdb local")

lmdb_path = os.getenv("SLURM_TMPDIR") + "/TextImages.lmdb"

for cpus_number in cpus_options:
    print("-----start index:" + str(start_index))
    ## generate updated key_list
    key_list = [f"{i:07}" for i in list(range(start_index+1, start_index+N_to_read+1))]
    
    ## split key_list to chunks
    n = math.ceil(len(key_list)/cpus_number)
    key_list_chunks = [key_list[i:i + n] for i in range(0, len(key_list), n)]     
    
    tic = time.time()
    res = Parallel(n_jobs=cpus_number, prefer = "processes", verbose=5)(delayed(read_fig_lmdb)(key_list) \
                                                                        for key_list in key_list_chunks )
    toc = time.time()
    timing_dict["lmdb_local"].append(toc-tic)
    #start_index = start_index + N_to_read

## flatten res 
tic_f = time.time()
res = [val for sublist in res for val in sublist]
print("flatten time for lmdb:" + str(time.time()-tic_f))

print("check loaded data")
print(res[1])

    
####################
##  HDF5 function ##
####################

def read_fig_hdf5(key_sublist):
  #print("work with file " + hdf5_path)
  ret_ar = []
  key_sublist.sort()
  with h5py.File(hdf5_path, 'r') as f:
    dset = f['images']
    chosen_stored_images = dset[key_sublist]
  return(chosen_stored_images)

####################
##  HDF5 scratch  ##
####################

print("read hdf5 scratch")

hdf5_path = '/scratch/work/public/datasets/TextRecognitionData_VGG_Oxford/hdf5/TextImages.hdf5'

for cpus_number in cpus_options:
    print("-----start index:" + str(start_index))
    ## generate updated key_list
    key_list_hdf5 = list(range(start_index, start_index+N_to_read))
    
    ## split key_list to chunks
    n = math.ceil(len(key_list_hdf5)/cpus_number)
    key_list_chunks = [key_list_hdf5[i:i + n] for i in range(0, len(key_list_hdf5), n)]     
    
    tic = time.time()
    res = Parallel(n_jobs=cpus_number, prefer = "processes", verbose=5)(delayed(read_fig_hdf5)(key_sublist) \
                                                                        for key_sublist in key_list_chunks)
    toc = time.time()
    timing_dict["hdf5"].append(toc-tic)
    #start_index = start_index + N_to_read

## flatten res 
tic_f = time.time()
res = [val for sublist in res for val in sublist]
print("flatten time for hdf5:" + str(time.time()-tic_f))

print("check loaded data")
print(res[1])

####################
##  HDF5 local    ##
####################

print("read hdf5 local")

hdf5_path = os.getenv("SLURM_TMPDIR") + "/TextImages.hdf5"

for cpus_number in cpus_options:
    print("-----start index:" + str(start_index))
    ## generate updated key_list
    key_list_hdf5 = list(range(start_index, start_index+N_to_read))
    
    ## split key_list to chunks
    n = math.ceil(len(key_list_hdf5)/cpus_number)
    key_list_chunks = [key_list_hdf5[i:i + n] for i in range(0, len(key_list_hdf5), n)]     
    
    tic = time.time()
    res = Parallel(n_jobs=cpus_number, prefer = "processes", verbose=5)(delayed(read_fig_hdf5)(key_sublist) \
                                                                        for key_sublist in key_list_chunks)
    toc = time.time()
    timing_dict["hdf5_local"].append(toc-tic)
    #start_index = start_index + N_to_read

## flatten res 
tic_f = time.time()
res = [val for sublist in res for val in sublist]
print("flatten time for hdf5:" + str(time.time()-tic_f))

print("check loaded data")
print(res[1])

###################################################################
## Prepare to read from disk: Subset paths to read files from    ##
###################################################################
print("read jpg from disk")

#############################
##  STURM_TMPDIR function  ##
#############################

def read_fig_disk(paths_to_read_sublist):
  ret_ar = []
  for p in paths_to_read_sublist:
      with open(p, mode='rb') as file:
            stored_image = file.read()   
            ret_ar.append(stored_image)
  return(ret_ar)

#############################
##  STURM_TMPDIR read      ##
#############################

print("read SLURM_TMPDIR local")

## make a list of paths from data frame column
paths_hdf5_all = df_hdf5["path"].tolist()

for cpus_number in cpus_options:
    print("-----start index:" + str(start_index))
    tic1 = time.time()
    ###### gets subset of paths
    paths_to_read = paths_hdf5_all[start_index:(start_index+N_to_read)]
    paths_to_read_full = [os.environ['SLURM_TMPDIR']  + "/mnt/ramdisk/max/90kDICT32px/" + s for s in paths_to_read]
    toc1 = time.time()
    timing_dict["subset_of_paths"].append(toc1-tic1)

    
    ## split key_list to chunks
    n = math.ceil(len(paths_to_read)/cpus_number)
    paths_to_read_chunks = [paths_to_read_full[i:i + n] for i in range(0, len(paths_to_read_full), n)]     
    
    tic = time.time()
    res = Parallel(n_jobs=cpus_number, prefer = "processes", verbose=5)(delayed(read_fig_disk)(paths_to_read_sublist) \
                                                                        for paths_to_read_sublist in paths_to_read_chunks)
    toc = time.time()
    timing_dict["SLURM_TMPDIR"].append(toc-tic)
    #start_index = start_index + N_to_read

## flatten res 
tic_f = time.time()
res = [val for sublist in res for val in sublist]
print("flatten time for lmdb:" + str(time.time()-tic_f))

print("check loaded data")
print(res[1])


######################
##  Display resutls ##
######################

#######################################################
import sys
#print("size or loaded data: " + str(sys.getsizeof(res)))
## size or loaded data: 1730440
## overhead ~100MB per process

print(timing_dict)    


cpus_options = timing_dict["cpus_options"]
del timing_dict["cpus_options"]
del timing_dict["subset_of_paths"]

df = pd.DataFrame(timing_dict, index=cpus_options)
df = df.round(3)

print(df)

## remove fist row
df = df.iloc[1:]

lines = df.plot.line(style='.-', markersize = 20)
lines.set_title("Reading time depending on number of cores used")
lines.set_xlabel("Number of cores used");
lines.set_ylabel("Time for reading (s)");
plt.savefig('./read_parallel.png')

#lines.set_xscale('log')
lines.set_yscale('log')
plt.savefig('./read_parallel_log.png')


