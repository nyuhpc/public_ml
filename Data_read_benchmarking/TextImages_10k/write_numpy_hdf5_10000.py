import re
import lmdb
import glob
import os
import os.path
from PIL import Image
import sqlite3
import time
import h5py
import io
import numpy as np


data_dir = "data_10000" + "/mnt/ramdisk/max/90kDICT32px/"

level_0 = glob.glob(os.path.join(data_dir + "/*"))

if len(level_0) == 0:
  raise ValueError("No files in directoy" + data_dir)

filename = "hdf5-numpy-10000.hdf5"
if os.path.exists(filename): os.remove(filename)
f = h5py.File(filename)
dt = h5py.special_dtype(vlen=np.dtype('uint8'))
dset = f.create_dataset('images', (10000, ), dtype=dt)

################################# SQL file with meta info
### sqlite file
filename = "hdf5-numpy-10000.sqlite"
if os.path.exists(filename): os.remove(filename)
sql_conn = sqlite3.connect(filename)
cur = sql_conn.cursor()
sql_table = 'CREATE TABLE IF NOT EXISTS meta (key integer PRIMARY KEY, label text NOT NULL, path text, shape0 integer, shape1 integer, shape2 integer);'
cur.execute(sql_table)

#################################
files_counter = 0

tic = time.time()

for L0 in level_0:
    # skip files (not directories)
    if os.path.isfile(L0):
      print("Not a directory on the first level: " + L0)
      continue
    
    level_1 = glob.glob(os.path.join(L0 + "/*"))

    for L1 in level_1:
        level_2 = glob.glob(os.path.join(L1 + "/*"))
        num_of_files = len(level_2)
        for i in range(num_of_files): 
              L2 = level_2[i]
              #print(L2)                 

              files_counter = files_counter + 1

              if(files_counter % 1000 == 0): 
                  print("working with file number:" + str(files_counter))
                             
              remaining_path, file_name = os.path.split(L2)
              remaining_path, dir1 = os.path.split(remaining_path)
              remaining_path, dir2 = os.path.split(remaining_path)
              original_path = dir2 + "/" + dir1 + "/" + file_name
              key = files_counter - 1 

              label = "".join(re.findall("_[a-zA-Z]+_", file_name)).strip("_")
              try:

                with open(L2, mode='rb') as file:
                    stored_image = file.read()
                PIL_image = Image.open(io.BytesIO(stored_image))
                value = np.asarray(PIL_image)
                    
                dset[files_counter - 1] = np.frombuffer(value, dtype='uint8')
                #dset[files_counter - 1] = pickle.dumps(value)
                
                
                sql_insert = 'INSERT INTO meta(key,label,path,shape0,shape1,shape2) VALUES(' + \
                              str(key) +',"' + label + '","' + original_path  + '",' + \
                              str(value.shape[0]) + ',' + str(value.shape[1]) + ',' + str(value.shape[2]) + ');'
                
                cur.execute(sql_insert)

              except Exception as e:
                print("Error observed: " + str(e))
                with open("errors_hdf5.txt", "a+") as file:
                  file.write("Error:" + str(e) + " . Observed for file " + L2 + "\n" )


sql_conn.commit()
sql_conn.close()
f.close()                  
print("Process took: " + str(time.time() - tic))
print("DONE!")

