from torch.utils.data import get_worker_info
from torch.utils.data import IterableDataset, DataLoader
import dicomo
import os

from compress_pickle import dump, load
import compress_pickle.utils

class PickleIterator(IterableDataset):

    def __init__(self, pickle_path, transform = None):
        assert isinstance(pickle_path, str), 'compressed_pickle_path is not a string'
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
            sample = next(self.pickle_stream)

            if self.transform:
                sample = self.transform(sample)

            return sample
        except:
            raise StopIteration


# testing code for PickleIterator
#origin = os.path.join('examples', 'dicom', 'CT')
#destination = os.path.join('examples', 'compressed', 'CT/')
#dicomo.compress_dicom_files(origin, destination)


database_path = os.path.join('examples', 'compressed', 'CT', '000001')
pickle_iter = iter(PickleIterator(database_path))

dicomo.plot_dicomo(next(pickle_iter))
dicomo.plot_dicomo(next(pickle_iter))
dicomo.plot_dicomo(next(pickle_iter))



