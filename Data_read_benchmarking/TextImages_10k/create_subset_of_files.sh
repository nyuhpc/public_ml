## data from
## Text Recognition Data (VGG, Oxford)
# https://www.robots.ox.ac.uk/~vgg/data/text/

## Do on scratch
## Dont extract the full data set - just enough: stop early to not hit quota of files nubmer
tar -C data_part -xf mjsynth.tar.gz  

# number of files to choose
N_files=10000
######################################
## copy selected number of files

rm -rf "data_${N_files}"
mkdir "data_${N_files}"
cat all_files.txt | head -n ${N_files} > selected_files
## make dirs
while IFS= read -r line; do mkdir -p $(echo $line | sed 's/[0-9_a-zA-Z.]*$//' | sed "s/^data_part/data_${N_files}/") ; done < selected_files
## copy files
while IFS= read -r line; do echo $line; cp $line $(echo "$line" | sed "s/^data_part/data_${N_files}/"); done < selected_files
#check
find data_${N_files}/* -type f | wc -l

tar -czvf data_${N_files}.tar.gz data_${N_files}




## Alternative - much slower version
############################################################
# get full list of files
tar -ztvf mjsynth.tar.gz > all.txt

# select N jpg files
N_files=10
cat all.txt | grep ".jpg$" | head -n $N_files | grep -Eo '[a-zA-Z0-9/._\-]+.jpg$' &> selected_jpg.txt

# extract only selected jpg files
tar -C data_${N_files} -zxvf <tar filename> <file you want to extract>

## extract selected files
mkdir "data_${N_files}"
while IFS= read -r line; do echo $line; tar -C data_${N_files} -zxvf mjsynth.tar.gz $line; done < selected_jpg.txt

