from timeit import timeit

cutoffs = [10, 100, 1000, 10000, 20000, 40000, 60000, 80000, 100000]
read_many_timings = {"disk": [], "lmdb": [], "hdf5": []}

_read_many_funcs = dict(                                           
    disk=read_many_disk, lmdb=read_many_lmdb, hdf5=read_many_hdf5
)                                                                  

for cutoff in cutoffs:
    for method in ("disk", "lmdb", "hdf5"):
        t = timeit(
            "_read_many_funcs[method](num_images)",
            setup="num_images=cutoff",
            number=1,
            globals=globals(),
        )
        read_many_timings[method].append(t)

        # Print out the method, cutoff, and elapsed time
        print(f"Method: {method}, No. images: {cutoff}, Time usage: {t}")

