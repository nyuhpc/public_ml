from random import sample
from timeit import timeit

#cutoffs = [10, 100, 1000, 10000, 20000, 40000, 60000, 80000, 100000]
cutoffs = [10, 100, 1000] + list(range(5000, 105000, 5000))
read_many_random_timings = {"disk": [], "lmdb": [], "hdf5": []}

num_images = 100000
image_indexes=list(range(num_images))

_read_many_random_funcs = dict(
    disk=read_many_random_disk, lmdb=read_many_random_lmdb, hdf5=read_many_random_hdf5
)


chosen_images = {}
for cutoff in cutoffs:
    chosen_images_list = sample(image_indexes, cutoff)
    chosen_images_list.sort()
    chosen_images[str(cutoff)] = chosen_images_list

for cutoff in cutoffs:
    for method in ("disk", "lmdb", "hdf5"):
        t = timeit(
            "_read_many_random_funcs[method](num_images_choose, num_images, chosen_images)",
            setup="num_images_choose=cutoff",
            number=1,
            globals=globals(),
        )
        read_many_random_timings[method].append(t)

        # Print out the method, cutoff, and elapsed time
        print(f"Method: {method}, No. images: {cutoff}, Time usage: {t}")

