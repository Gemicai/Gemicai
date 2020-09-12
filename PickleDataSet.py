from torch.utils.data import get_worker_info
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import dicomo
import os
import torch

from compress_pickle import dump, load
import compress_pickle.utils

class PickleDataSet(IterableDataset):

    def __init__(self, pickle_path, dicomo_fields, transform = None):
        assert isinstance(dicomo_fields, list), 'dicomo_fields is not a list'
        assert isinstance(pickle_path, str), 'pickle_path is not a string'
        self.dicomo_fields = dicomo_fields
        self.pickle_path = pickle_path
        self.transform = transform

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            self.pickle_stream = dicomo.stream_pickles(self.pickle_path)
            return self
        else:
            raise Exception("PickleDataSet does not support multi-process data loading")

    def __next__(self):
        try:
            # get next dicomo class from the stream
            dicomo_class = next(self.pickle_stream)

            # fetch values of the fields we are interested in
            field_list = []
            for field in self.dicomo_fields:
                try:
                    # ugly but works
                    # check if transform is specified and if it should be applied
                    temp = getattr(dicomo_class, field)
                    if self.transform and field == 'tensor':
                        try:
                            temp = self.transform(temp)
                        except:
                            raise Exception('Could not apply specified transformation to the dicom image')
                    field_list.append(temp)
                except:
                    None

            return field_list
        except:
            raise StopIteration


def print_labels_and_display_images(tensors, labels):
    for index, tensor in enumerate(tensors):
        print(labels[index])
        dicomo.plt.imshow(tensor, cmap='gray')
        dicomo.plt.show()


"""
# testing code for the PickleDataSet
#origin = os.path.join('examples', 'dicom', 'CT')
#destination = os.path.join('examples', 'compressed', 'CT/')
#dicomo.compress_dicom_files(origin, destination)

# example usage of the PickleDataSet
database_path = os.path.join('examples', 'compressed', 'CT', '000001')

# while creating PickleDataSet we pass a path to a pickle that hold the data
# and a list of the fields that we want to extract from the dicomo object
pickle_iter = PickleDataSet(database_path, ['tensor', 'bpe'])
trainloader = torch.utils.data.DataLoader(pickle_iter, batch_size=4, shuffle=False, num_workers=0)

# get new a batch
dataiter = iter(trainloader)
tensors, labels = dataiter.next()

# display the batch
print_labels_and_display_images(tensors, labels)
"""