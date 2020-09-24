from torch.utils.data import get_worker_info
from torch.utils.data import IterableDataset
from abc import ABC, abstractmethod
import os

from gemicai import dicomo


class ABCIterator(ABC, IterableDataset):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def is_pinned(self):
        pass


class PickledDicomoDataFolder(ABCIterator):
    def __init__(self, base_path, dicomo_fields, transform=None, pin_memory=False):
        assert isinstance(dicomo_fields, list), 'dicomo_fields is not a list'
        assert isinstance(base_path, str), 'base_path is not a string'
        self.dicomo_fields = dicomo_fields
        self.pin_memory = pin_memory
        self.base_path = base_path
        self.transform = transform
        self.len = 0

    def __iter__(self):
        self.data_set_gen = self.get_next_data_set()
        self.data_set = next(self.data_set_gen)
        self.len = 0
        return self

    def __next__(self):
        try:
            while True:
                try:
                    temp = next(self.data_set)
                    self.len += 1
                    return temp
                except:
                    self.data_set = next(self.data_set_gen)
        except:
            raise StopIteration

    def __len__(self):
        return self.len

    def get_next_data_set(self):
        for root, dirs, files in os.walk(self.base_path):
            for name in files:
                yield iter(PickledDicomoDataSet(os.path.join(root, name), self.dicomo_fields, self.transform, self.pin_memory))
        raise StopIteration

    def is_pinned(self):
        return self.pin_memory


class PickledDicomoDataSet(ABCIterator):

    def __init__(self, pickle_path, dicomo_fields, transform=None, pin_memory=False):
        assert isinstance(dicomo_fields, list), 'dicomo_fields is not a list'
        assert isinstance(pickle_path, str), 'pickle_path is not a string'
        self.dicomo_fields = dicomo_fields
        self.pickle_path = pickle_path
        self.pin_memory = pin_memory
        self.transform = transform
        self.len = 0

    def __iter__(self):
        worker_info = get_worker_info()
        self.len = 0

        if worker_info is None:
            self.pickle_stream = self.stream_pickled_dicomos()
            return self
        else:
            raise Exception("PickledDicomoDataSet does not support multi-process data loading")

    def __next__(self):
        try:
            # get next dicomo class from the stream
            dicomo_class = next(self.pickle_stream)

            # fetch values of the fields we are interested in
            field_list = []
            for field in self.dicomo_fields:
                try:
                    temp = getattr(dicomo_class, field)

                    # check if transform is specified and if it should be applied
                    if self.transform is not None and field == 'tensor':
                        try:
                            temp = self.transform(temp)
                        except:
                            raise Exception('Could not apply specified transformation to the dicom image')

                    # pin (page-lock) memory, it allows us to use asynchronous GPU copies
                    if self.pin_memory:
                        temp = temp.pin_memory()

                    field_list.append(temp)
                except:
                    None

            self.len += 1
            return field_list
        except:
            raise StopIteration

    def __len__(self):
        return self.len

    def stream_pickled_dicomos(self):
        tmp = dicomo.tempfile.NamedTemporaryFile(mode="ab+", delete=False)
        try:
            dicomo.unzip_to_file(tmp, self.pickle_path)
            while True:
                yield dicomo.pickle.load(tmp)
        except EOFError:
            pass
        finally:
            tmp.close()
            os.remove(tmp.name)

    def is_pinned(self):
        return self.pin_memory
