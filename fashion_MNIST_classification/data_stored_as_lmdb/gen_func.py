def lmdb_data_generator(env_x, y_labels, batch_size = 32, randomize = False):
    import sys
    import numpy as np
    import lmdb
    import pickle
    import random
    from tensorflow.keras.utils import to_categorical

    data_length = y_labels.shape[0]
    
    all_indexes = list(range(data_length))
    with env_x.begin(write=False) as lmdb_txn:
        while True:
            labels = []

            ## make sure generator always produces data
            ## Issue to address: if only a couple of images are remaining in current list
            ## on the next call only this couple will be returned
            if(len(all_indexes) < batch_size): 
                all_indexes = list(range(data_length))

            if randomize:
                selected_indexes = random.sample(all_indexes, batch_size)
            else:
                selected_indexes = all_indexes[:batch_size]

            for x in selected_indexes: 
                all_indexes.remove(x)
            
            batch_array = np.zeros((batch_size, 28, 28))
            batch_index = 0

            for file_index in selected_indexes:
                key = f"{file_index:07}"
                stored_data = lmdb_txn.get(key.encode())
                batch_array[batch_index] = pickle.loads(stored_data)                
                batch_index = batch_index + 1

                ## labels
                labels.append(y_labels[file_index])

            # return values
            labels = to_categorical(labels, num_classes = 10)
            yield(batch_array, labels)

## check
#env_train = lmdb.open("x-numpy-train.lmdb")
#y_train = pickle.load( open( "y_train.pkl", "rb" ) )
# gen_train = lmdb_data_generator(
#    env_x = env_train, y_labels = y_train, batch_size = 32)
# next(gen_train)
# im, lab = next(gen_train)
# im.shape
# lab.shape
