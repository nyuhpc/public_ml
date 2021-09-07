import re
import lmdb
import cv2
import glob
import os
import os.path
from PIL import Image
import sqlite3
import time

## Check SLURM is running
if os.environ.get("SLURM_TMPDIR") == None:
   raise ValueError("STLUM_TMPDIR not defined")
data_dir = os.environ.get("SLURM_TMPDIR") + "/mnt/ramdisk/max/90kDICT32px/"

## Upack files to SLURM_TMPDIR
print("untar files to $SLURM_TMPDIR")
os.system("tar -C $SLURM_TMPDIR -xf mjsynth.tar.gz")
print("untar done")

## List files inside the first level directory
level_0 = glob.glob(os.path.join(data_dir + "/*"))

if len(level_0) == 0:
  raise ValueError("No files in directoy" + data_dir)

################################# LMDB file
env = lmdb.open("TextImages.lmdb", map_size=int(100e9))

################################# SQL file with meta info
### sqlite file
sql_conn = sqlite3.connect("TextImages.sqlite")
sql_table = 'CREATE TABLE IF NOT EXISTS meta (key text PRIMARY KEY, label text NOT NULL, path text);'
cur = sql_conn.cursor()
cur.execute(sql_table)

#################################
file_counter = 0

tic = time.time()

for L0 in level_0:
  with env.begin(write=True) as lmdb_txn: ## commit transaction for every top sub-directory
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

              file_counter = file_counter + 1

              if(file_counter % 1000 == 0): 
                  print(lmdb_txn.stat()) 
                  print("working with file number:" + str(file_counter))
                             
              remaining_path, file_name = os.path.split(L2)
              remaining_path, dir1 = os.path.split(remaining_path)
              remaining_path, dir2 = os.path.split(remaining_path)
              original_path = dir2 + "/" + dir1 + "/" + file_name
              
              ## use 7 digits to cover < 10 mil files
              key = f"{file_counter:07}"
              
              label = "".join(re.findall("_[a-zA-Z]+_", file_name)).strip("_")
              try:

                with open(L2, mode='rb') as file: # b is important -> binary
                    value = file.read()
                
                lmdb_txn.put(key.encode(), value)  

                sql_insert = 'INSERT INTO meta(key,label,path) VALUES(' + \
                              '"' + key +'","' + label + '","' + original_path  + '");'
                
                cur.execute(sql_insert)

              except Exception as e:
                print("Error observed: " + str(e))
                with open("errors_lmdb.txt", "a+") as file:
                  file.write("Error:" + str(e) + " . Observed for file " + L2 + "\n" )
  
  print("------commit to lmdb------")

sql_conn.commit()
sql_conn.close()
env.close()

print("Process took: " + str(time.time() - tic))
print("DONE!")
