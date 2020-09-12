from torch.utils.data import get_worker_info
from torch.utils.data import IterableDataset, DataLoader
import dicomo
import os

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
            raise Exception("PickleIterator does not support multi-process data loading")

    def __next__(self):
        try:
            # get next dicomo class from the stream
            dicomo_class = next(self.pickle_stream)

            # fetch values of the fields we are interested in
            dictionary = {}
            for field in self.dicomo_fields:
                try:
                    dictionary[field] = getattr(dicomo_class, field)
                except:
                    None

            # if we have specified a transform try to apply it
            if self.transform:
                try:
                    dictionary['tensor'] = self.transform(dictionary['tensor'])
                except:
                    None

            return dictionary
        except:
            raise StopIteration


# testing code for PickleIterator
#origin = os.path.join('examples', 'dicom', 'CT')
#destination = os.path.join('examples', 'compressed', 'CT/')
#dicomo.compress_dicom_files(origin, destination)


database_path = os.path.join('examples', 'compressed', 'CT', '000001')
pickle_iter = iter(PickleDataSet(database_path, ['bpe', 'tensor']))

print(next(pickle_iter))
print(next(pickle_iter))
print(next(pickle_iter))
print(next(pickle_iter))



