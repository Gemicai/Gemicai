from torch.utils.data import get_worker_info
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
import gemicai as gem
import math
import os

# This class interface serves as a basis for any data iterator
class GemicaiDataset(ABC, IterableDataset):
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


# This class interface serves as a basis for any dicomo data iterator
class DicomoDataset(GemicaiDataset):
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

    # Loads classifier object from .pkl file
    @staticmethod
    def from_config(dataset_config):
        if dataset_config['type'] == PickledDicomoDataFolder:
            return PickledDicomoDataFolder(base_path=dataset_config['path'],
                                           dicomo_fields=dataset_config['object_fields'],
                                           transform=dataset_config['transform'],
                                           constraints=dataset_config['constraints'])
        if dataset_config['type'] == ConcurrentPickledDicomoTaskSplitter:
            return ConcurrentPickledDicomoTaskSplitter(base_path=dataset_config['path'],
                                                       dicomo_fields=dataset_config['object_fields'],
                                                       transform=dataset_config['transform'],
                                                       constraints=dataset_config['constraints'])

    @staticmethod
    def from_file(file_path, dicomo_fields=[], transform=None, constraints={}):
        if not os.path.isfile(file_path):
            raise FileNotFoundError
        return PickledDicomoDataSet(GemicaiDataset, dicomo_fields, transform, constraints)

    @staticmethod
    def from_folder(folder_path, dicomo_fields=[], transform=None, constraints={}):
        if not os.path.isdir(folder_path):
            raise NotADirectoryError
        return ConcurrentPickledDicomoTaskSplitter(folder_path, dicomo_fields, transform, constraints)

    @staticmethod
    def get_dicomo_data_set(data_set_path, dicomo_fields=[], constraints={}):
        transform = gem.torchvision.transforms.Compose([
            gem.torchvision.transforms.ToPILImage(),
            gem.torchvision.transforms.Grayscale(3),
            gem.torchvision.transforms.ToTensor()
        ])

        if os.path.isfile(data_set_path):
            # while creating PickleDataSet we pass a path to a pickle that hold the data
            # and a list of the fields that we want to extract from the dicomo object
            return DicomoDataset.from_file(data_set_path, dicomo_fields, transform)
        else:
            return DicomoDataset.from_folder(data_set_path, dicomo_fields, transform)


class ConcurrentPickledDicomoTaskSplitter(DicomoDataset):
    def __init__(self, base_path, dicomo_fields, transform=None, constraints={}):
        assert isinstance(dicomo_fields, list), 'dicomo_fields is not a list'
        assert isinstance(base_path, str), 'base_path is not a string'
        self.dicomo_fields = dicomo_fields
        self.constraints = constraints
        self.base_path = base_path
        self.transform = transform
        self.len = 0

        self.file_pool = []
        for root, dirs, files in os.walk(self.base_path):
            for name in files:
                self.file_pool.append(os.path.join(root, name))

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            # we are in a single threaded environment so there is no need to modify the data set
            return iter(PickledDicomoFilePool(self.file_pool, self.dicomo_fields, self.transform, self.constraints))
        else:
            # we are in a multi-threaded environment, slice the dataset
            per_worker = int(math.ceil(len(self.file_pool) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.file_pool))
            return iter(PickledDicomoFilePool(self.file_pool[start:end],
                                              self.dicomo_fields, self.transform, self.constraints))

    def __next__(self):
        raise Exception("This 'Iterator' is meant to split a file pool and return PickledDicomoFilePool")

    def __len__(self):
        raise Exception("This 'Iterator' is meant to split a file pool and return PickledDicomoFilePool")

    def can_be_parallelized(self):
        return True


class PickledDicomoFilePool(DicomoDataset):
    def __init__(self, file_pool, dicomo_fields, transform=None, constraints={}):
        assert isinstance(dicomo_fields, list), 'dicomo_fields is not a list'
        assert isinstance(file_pool, list), 'file_pool is not a string'
        self.dicomo_fields = dicomo_fields
        self.constraints = constraints
        self.file_pool = file_pool
        self.transform = transform
        self.set_generator = None
        self.data_set = None
        self.len = 0

    def __iter__(self):
        self.set_generator = None
        self.data_set = None
        return self

    def __next__(self):
        try:
            while True:
                try:
                    temp = next(self.data_set)
                    self.len += 1
                    return temp
                except:
                    if self.set_generator is None:
                        self.set_generator = self.pool_walker()
                    self.data_set = next(self.set_generator)
        except:
            raise StopIteration

    def __len__(self):
        return self.len

    def can_be_parallelized(self):
        return False

    def pool_walker(self):
        for file_path in self.file_pool:
            yield iter(PickledDicomoDataSet(file_path, self.dicomo_fields, self.transform, self.constraints))
        raise StopIteration


class PickledDicomoDataFolder(DicomoDataset):
    def __init__(self, base_path, dicomo_fields, transform=None, constraints={}):
        assert isinstance(dicomo_fields, list), 'dicomo_fields is not a list'
        assert isinstance(base_path, str), 'base_path is not a string'
        self.dicomo_fields = dicomo_fields
        self.base_path = base_path
        self.transform = transform
        self.constraints = constraints
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
        cnt = gem.LabelCounter()
        for data in DataLoader(self, 4, shuffle=False):
            for label in data[self.dicomo_fields.index(count_field)]:
                cnt.update(label)
        return cnt

    def get_next_data_set(self):
        for root, dirs, files in os.walk(self.base_path):
            for name in files:
                yield iter(PickledDicomoDataSet(os.path.join(root, name), self.dicomo_fields, self.transform,
                                                self.constraints))
        raise StopIteration

    def can_be_parallelized(self):
        return False


class PickledDicomoDataSet(DicomoDataset):
    def __init__(self, pickle_path, dicomo_fields, transform=None, constraints={}):
        assert isinstance(dicomo_fields, list), 'dicomo_fields is not a list'
        assert isinstance(pickle_path, str), 'pickle_path is not a string'
        self.dicomo_fields = dicomo_fields
        self.pickle_path = pickle_path
        self.transform = transform
        self.constraints = constraints
        self.len = 0

    def __iter__(self):
        self.len = 0
        self.pickle_stream = self.stream_pickled_dicomos()
        return self

    def __next__(self):
        try:
            # get next dicomo class from the stream
            dicomo_class = next(self.pickle_stream)
            # constraints is a dictionary.
            for k in self.constraints.keys():
                if self.constraints[k] != getattr(dicomo_class, k):
                    return self.__next__()
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
        tmp = gem.tempfile.NamedTemporaryFile(mode="ab+", delete=False)
        try:
            gem.unzip_to_file(tmp, self.pickle_path)
            while True:
                yield gem.pickle.load(tmp)
        except EOFError:
            pass
        finally:
            tmp.close()
            os.remove(tmp.name)

    def can_be_parallelized(self):
        return False
