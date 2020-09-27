from torch.utils.data import get_worker_info
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from gemicai import dicomo
import os


class GEMICAIABCIterator(ABC, IterableDataset):
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
    def can_be_parallelized(self):
        pass


class PickledDicomoDataFolder(GEMICAIABCIterator):
    def __init__(self, base_path, dicomo_fields, transform=None):
        assert isinstance(dicomo_fields, list), 'dicomo_fields is not a list'
        assert isinstance(base_path, str), 'base_path is not a string'
        self.dicomo_fields = dicomo_fields
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

    def __str__(self):
        return str(self.summaray(count_field=self.dicomo_fields[1]))

    def summaray(self, count_field=None):
        assert count_field is not None, 'Specify which field you want to summarize: {}'.format(self.dicomo_fields)
        cnt = dicomo.LabelCounter()
        for data in DataLoader(self, 4, shuffle=False):
            for label in data[self.dicomo_fields.index(count_field)]:
                cnt.update(label)
        return cnt

    def get_next_data_set(self):
        for root, dirs, files in os.walk(self.base_path):
            for name in files:
                yield iter(PickledDicomoDataSet(os.path.join(root, name), self.dicomo_fields, self.transform))
        raise StopIteration

    def can_be_parallelized(self):
        return False


class PickledDicomoDataSet(GEMICAIABCIterator):

    def __init__(self, pickle_path, dicomo_fields, transform=None):
        assert isinstance(dicomo_fields, list), 'dicomo_fields is not a list'
        assert isinstance(pickle_path, str), 'pickle_path is not a string'
        self.dicomo_fields = dicomo_fields
        self.pickle_path = pickle_path
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

    def can_be_parallelized(self):
        return False
