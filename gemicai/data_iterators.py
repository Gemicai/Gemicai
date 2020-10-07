from torch.utils.data import get_worker_info
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
import gemicai.data_objects
import gemicai as gem
import math
import os
import matplotlib.pyplot as plt

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

    @staticmethod
    def from_file(file_path, labels=[], transform=None, constraints={}):
        if not os.path.isfile(file_path):
            raise FileNotFoundError
        return PickledDicomoDataSet(file_path, labels, transform, constraints)

    @staticmethod
    def from_directory(folder_path, labels=[], transform=None, constraints={}):
        if not os.path.isdir(folder_path):
            raise NotADirectoryError
        return ConcurrentPickledDicomoTaskSplitter(folder_path, labels, transform, constraints)

    @staticmethod
    def get_dicomo_dataset(data_set_path, labels=[], constraints={}):
        transform = gem.torchvision.transforms.Compose([
            gem.torchvision.transforms.ToPILImage(),
            gem.torchvision.transforms.Grayscale(3),
            gem.torchvision.transforms.ToTensor()
        ])

        if os.path.isfile(data_set_path):
            # while creating PickleDataSet we pass a path to a pickle that hold the data
            # and a list of the fields that we want to extract from the dicomo object
            return DicomoDataset.from_file(data_set_path, labels, transform, constraints)
        else:
            return DicomoDataset.from_directory(data_set_path, labels, transform, constraints)


class ConcurrentPickledDicomoTaskSplitter(DicomoDataset):
    def __init__(self, base_path, labels, transform=None, constraints={}):
        assert isinstance(labels, list), 'dicomo_fields is not a list'
        assert isinstance(base_path, str), 'base_path is not a string'
        self.labels = labels
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
            return iter(PickledDicomoFilePool(self.file_pool, self.labels, self.transform, self.constraints))
        else:
            # we are in a multi-threaded environment, slice the dataset
            per_worker = int(math.ceil(len(self.file_pool) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, len(self.file_pool))
            return iter(PickledDicomoFilePool(self.file_pool[start:end],
                                              self.labels, self.transform, self.constraints))

    def __next__(self):
        raise Exception("This 'Iterator' is meant to split a file pool and return PickledDicomoFilePool")

    def __len__(self):
        raise Exception("This 'Iterator' is meant to split a file pool and return PickledDicomoFilePool")

    def can_be_parallelized(self):
        return True

    def subset(self, constraints):
        return ConcurrentPickledDicomoTaskSplitter(self.base_path, self.labels, self.transform,
                                                   {**self.constraints, **constraints})

    def classes(self, label):
        return list(self.summarize(label, print_summary=False).dic.keys())

    def summarize(self, label, constraints={}, print_summary=True):
        temp = self.labels
        self.labels = []
        cnt = gem.LabelCounter()
        constraints = {**self.constraints, **constraints}
        for dicomo in self:
            if dicomo.meets_constraints(constraints):
                cnt.update(dicomo.get_value_of(label))
        self.labels = temp
        if print_summary:
            print(cnt)
        else:
            return cnt


class PickledDicomoFilePool(DicomoDataset):
    def __init__(self, file_pool, labels, transform=None, constraints={}):
        assert isinstance(labels, list), 'dicomo_fields is not a list'
        assert isinstance(file_pool, list), 'file_pool is not a string'
        self.labels = labels
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
            yield iter(PickledDicomoDataSet(file_path, self.labels, self.transform, self.constraints))
        raise StopIteration


class PickledDicomoDataFolder(DicomoDataset):
    def __init__(self, base_path, labels, transform=None, constraints={}):
        assert isinstance(labels, list), 'dicomo_fields is not a list'
        assert isinstance(base_path, str), 'base_path is not a string'
        self.labels = labels
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
        return str(self.summaray(count_field=self.labels[1]))

    def summaray(self, count_field=None):
        assert count_field is not None, 'Specify which field you want to summarize: {}'.format(self.labels)
        cnt = gem.LabelCounter()
        for data in DataLoader(self, 4, shuffle=False):
            for label in data[self.labels.index(count_field)]:
                cnt.update(label)
        return cnt

    def get_next_data_set(self):
        for root, dirs, files in os.walk(self.base_path):
            for name in files:
                yield iter(PickledDicomoDataSet(os.path.join(root, name), self.labels, self.transform,
                                                self.constraints))
        raise StopIteration

    def can_be_parallelized(self):
        return False


class PickledDicomoDataSet(DicomoDataset):
    def __init__(self, pickle_path, labels=[], transform=None, constraints={}):
        assert isinstance(labels, list), 'dicomo_fields is not a list'
        assert isinstance(pickle_path, str), 'pickle_path is not a string'
        self.labels = labels
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
            if not isinstance(dicomo_class, gemicai.data_objects.DicomObject):
                raise Exception("pickled dataset should contain gemicai.data_iterators.DicomObject but it contains "
                                + type(dicomo_class))

            if len(self.labels) == 0:
                return dicomo_class

            # constraints is a dictionary.
            for k in self.constraints.keys():
                if self.constraints[k] != dicomo_class.get_value_of(k):
                    return self.__next__()

            # fetch a tensor
            tensor = dicomo_class.tensor

            # check if transform is specified if yes apply it
            if self.transform is not None:
                try:
                    tensor = self.transform(tensor)
                except:
                    raise Exception('Could not apply specified transformation to the dicom image')

            labels = []
            # fetch values of the labels we are interested in
            for label in self.labels:
                labels.append(dicomo_class.get_value_of(label))

            self.len += 1

            # All hail to python for not returning reference to the temp object after calling .append
            # which forces me to do this
            return [tensor] + labels
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

    def subset(self, constraints):
        return PickledDicomoDataSet(self.pickle_path, self.labels, self.transform, {**self.constraints, **constraints})

    # All functions below here are a bit hacky
    def classes(self, label):
        return list(self.summarize(label, print_summary=False).dic.keys())

    def summarize(self, label, constraints={}, print_summary=True):
        temp = self.labels
        self.labels = []
        cnt = gem.LabelCounter()
        constraints = {**self.constraints, **constraints}
        for dicomo in self:
            if dicomo.meets_constraints(constraints):
                cnt.update(dicomo.get_value_of(label))
        self.labels = temp
        if print_summary:
            print(cnt)
        else:
            return cnt

    def plot_one_of_every(self, label, cmap='gray_r'):
        classes = self.classes(label)
        ooe = []
        temp = self.labels
        self.labels = []
        for dicomo in self:
            v = dicomo.get_value_of(label)
            if v in classes:
                ooe.append(dicomo)
                classes.remove(v)
            if len(classes) == 0:
                break
        self.labels = temp

        for d in ooe:
            s = ''
            for label, value in zip(d.label_types, d.labels):
                s += '{:15s} : {}\n'.format(label, value)
            print(s)
            plt.imshow(d.tensor, cmap)
            plt.show()

