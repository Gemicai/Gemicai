"""This module contains data iterators which are used in order to traverse a data set and retrieve a relevant
information from the DataObjects it contains."""

from torch.utils.data import get_worker_info
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import gemicai.data_objects
from itertools import count
import gemicai as gem
import math
import os


class GemicaiDataset(ABC, IterableDataset):
    """This interface class serves as a basis for the every Gemicai's data iterator."""

    @abstractmethod
    def __init__(self):
        """Should initialize the iterator, it could for example take in a path to the data set."""
        pass

    @abstractmethod
    def __iter__(self):
        """Should return a valid iterator object on which it is possible to call next."""
        pass

    @abstractmethod
    def __next__(self):
        """Should return a list with a next tensor and it's label."""
        pass

    @abstractmethod
    def __len__(self):
        """Should return a number of items which were iterated over so far."""
        pass

    @abstractmethod
    def summarize(self, label, print_summary=True):
        """Should return or print a summary of all the DataObject values in the data set selected by the label.

        :param label: field label which values to summarize, for example 'BodyPartExamined' or 'Modality'
        :type label: str
        :param print_summary: whenever to print or return an instance of gemicai.label_counters.GemicaiLabelCounter
            object
        :type print_summary: bool
        :return: if print_summary is set to false a class that extends a gemicai.label_counters.GemicaiLabelCounter
        """
        pass

    @abstractmethod
    def subset(self, constraints):
        """Should return a subset of a current data set.

        :param constraints: dictionary with a data set constraints, eg. {'Modality': 'CT'}
        :type constraints: dict
        :return: a correct user defined iterator type which extends gemicai.data_iterators.GemicaiDataset
        """
        pass

    @abstractmethod
    def classes(self, label):
        """Should return a list of all the classes in the data set.

        :param label: label to summarize on
        :type label: str
        :return: list of possible label values present in the data set
        """
        pass

    @abstractmethod
    def plot_one_of_every(self, label, cmap='gray_r'):
        """Should plot one image per class.

        :param label: label according to which we will look for a unique values
        :type label: str
        :param cmap: color scheme
        :type cmap: str
        """
        pass

    @abstractmethod
    def can_be_parallelized(self):
        """Should return a boolean specifying whenever current iterator supports parallelized resource loading.

        :return: True if parallelized resource loading is supported, False otherwise
        """
        pass

    @abstractmethod
    def save(self, directory):
        """Should save DataObjects fulfilling the iterator's constraints to the specified directory.

        :param directory: a valid directory path to which data sets will be saved
        :type directory: str
        """
        pass

    @abstractmethod
    def erase(self):
        """Should erase iterator's data sets from the file system."""
        pass

    @abstractmethod
    def modify(self, index, fields):
        """Should modify an underlying DataObject with a given index using a provided dictionary with a field-value
        mappings, eg. set.modify(2, {'Modality': 'CT'}) should set 'Modality' of a third object in the data set to 'CT'.
        Note that the first object in the data set should have an index equal to zero.

        :param index: a non-negative index of the object to modify
        :type  index: int
        :param fields: fields to modify, eg. {'Modality': 'CT', ...}
        :type  fields: dict
        """
        pass

    @abstractmethod
    def split(self, sets={'train': 0.8, 'test': 0.2}, max_objects_per_file=1000, self_erase_afterwards=False):
        """Should split the underlying original data set into N-sets specified by the sets parameter. Each data set that
        is a file with a .gemset extension should contain up to a max_objects_per_file DataObjects as a result. If
        self_erase_afterwards is set to True the original data set should be erased.

        :param sets: dictionary with a valid path-ratio mappings, eg.
            sets={'train': 0.8, 'test': 0.2} should split the original data set into two sets with a ratio of 8:2. First
            set should be saved into a 'train' folder and the second into a 'test' folder. The sum of all the ratios
            added up all together should be equal to 1.0.
        :type sets: dict
        :param max_objects_per_file: a non-negative integer specifying how many objects at maximum can be inside of each
            .gemset file
        :type max_objects_per_file: int
        :param self_erase_afterwards: specifies whenever the original data set should be removed after splitting or not.
        :type self_erase_afterwards: bool
        """
        pass


class DicomoDataset(GemicaiDataset):
    """Every provided non-abstract data iterator extends this class and calls it's __init__ method with a following
    argument:

    :param label_counter_type: label counter used by the summarize method
    :type label_counter_type: gemicai.label_counter.GemicaiLabelCounter
    :raises TypeError: raised if any of the parameters has an invalid type
    """

    def __init__(self, label_counter_type=gem.label_counters.LabelCounter):
        if not issubclass(label_counter_type, gem.label_counters.GemicaiLabelCounter):
            raise TypeError('label_counter_type should have a base class of gem.label_counters.GemicaiLabelCounter')
        self.lbl_ctr_tpe = label_counter_type

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    def __getitem__(self, arg):
        """Custom override for a __getitem__ method. Returns a new data iterator that returns only objects with a
        label by a given by an arg

        :param arg: index of a label stored in the self.labels list
        :type arg: int
        :return: a new data iterator of the same type
        :raises TypeError: raised if arg is not of an int type
        """
        if not isinstance(arg, int):
            raise TypeError("Argument should have an int type")

        arg = self.labels[arg]
        if arg not in self.labels:
            raise ValueError('Specified argument not in gemset labels. Valid labels are: {}'.format(self.labels))
        return type(self)(self.base_path, labels=[arg], transform=self.transform, constraints=self.constraints)

    @abstractmethod
    def can_be_parallelized(self):
        pass

    @abstractmethod
    def subset(self):
        pass
    
    @abstractmethod
    def save(self, directory):
        pass

    @abstractmethod
    def erase(self):
        pass

    @abstractmethod
    def modify(self, index, fields):
        pass

    @abstractmethod
    def split(self, sets={'train': 0.2, 'test': 0.8}, self_erase_afterwards=False):
        pass

    def classes(self, label):
        """Returns a list of all of the classes in the data set.

        :param label: label to summarize on
        :type label: str
        :return: list of possible label values present in the data set
        """
        return list(self.summarize(label, print_summary=False).dic.keys())

    def summarize(self, label, print_summary=True):
        """Returns or prints a summary of all the DataObject values in the data set selected by the label.

        :param label: field label which values to summarize, for example 'BodyPartExamined' or 'Modality'
        :type label: str
        :param print_summary: whenever to print or return an instance of gemicai.label_counters.GemicaiLabelCounter
            object
        :type print_summary: bool
        :return: if print_summary is set to false a class that extends a gemicai.label_counters.GemicaiLabelCounter
        :raise TypeError: raised whenever one of the parameter has an invalid type
        """
        if not isinstance(label, str):
            raise TypeError("label should be a string")
        if not isinstance(print_summary, bool):
            raise TypeError("print_summary should be a boolean")

        temp = self.labels
        self.labels = []
        cnt = self.lbl_ctr_tpe(label)
        for dicomo in self:
            cnt.update(dicomo.get(label))
        self.labels = temp
        if print_summary:
            print(cnt)
        else:
            return cnt

    def plot_one_of_every(self, label, cmap='gray_r'):
        """Plots one image per value type.

        :param label: label according to which we will look for a unique values, eg, 'Modality'
        :type label: str
        :param cmap: color scheme
        :type cmap: str
        :raises TypeError: raised whenever label is not a str
        """
        if not isinstance(label, str):
            raise TypeError("label should be a string")

        classes = self.classes(label)
        ooe = []
        temp = self.labels
        self.labels = []
        for dicomo in self:
            v = dicomo.get(label)
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

    @staticmethod
    def from_file(file_path, labels=[], transform=None, constraints={}):
        """Creates a data iterator for a supplied .gemset file

        :param file_path: a valid path to a .gemset file
        :type file_path: str
        :param labels: labels specifying which DataObject values except for a tensor will be returned by the next() call
        :type labels: Optional[list]
        :param transform: optional transforms to be applied on the tensor
        :type transform: Optional[any torchvision.transforms]
        :param constraints: optional constraints that the DataObject has to fulfil in order to be returned by the
            next() call, eg. {'Modality': 'CT'} or {'Modality': ['CT', 'MG']}
        :type constraints: Optional[dict]
        :return: a valid gemicai.data_iterators.PickledDicomoDataSet object
        :raises FileNotFoundError: raised whenever file_path does not point to any valid file
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError
        return PickledDicomoDataSet(file_path, labels, transform, constraints)

    @staticmethod
    def from_directory(folder_path, labels=[], transform=None, constraints={}):
        """Creates a data iterator from the supplied folder which should contain .gemset data sets

        :param folder_path: a valid path to an existing folder which contains .gemset data sets
        :type folder_path: str
        :param labels: labels specifying which DataObject values except for a tensor will be returned by the next() call
        :type labels: Optional[list]
        :param transform: optional transforms to be applied on the tensor
        :type transform: Optional[any torchvision.transforms]
        :param constraints: optional constraints that the DataObject has to fulfil in order to be returned by the
            next() call, eg. {'Modality': 'CT'} or {'Modality': ['CT', 'MG']}
        :type constraints: Optional[dict]
        :return: a valid gemicai.data_iterators.ConcurrentPickledDicomObjectTaskSplitter object
        :raises NotADirectoryError: raised whenever passed folder_path is invalid
        """
        if not os.path.isdir(folder_path):
            raise NotADirectoryError
        return ConcurrentPickledDicomObjectTaskSplitter(folder_path, labels, transform, constraints)

    @staticmethod
    def get_dicomo_dataset(data_set_path, labels=[], constraints={}):
        """Created a data iterator from the supplied file or folder path

        :param data_set_path:  a valid path to an existing folder which contains .gemset data sets
            or a valid path to a .gemset file
        :param labels: labels specifying which DataObject values except for a tensor will be returned by the next() call
        :type labels: Optional[list]
        :param constraints: optional constraints that the DataObject has to fulfil in order to be returned by the
            next() call, eg. {'Modality': 'CT'} or {'Modality': ['CT', 'MG']}
        :type constraints: Optional[dict]
        :return: gemicai.data_iterators.PickledDicomoDataSet object if file path was supplied otherwise
            gemicai.data_iterators.ConcurrentPickledDicomObjectTaskSplitter object
        :raises FileNotFoundError: raised whenever file_path does not point to any valid file
        :raises NotADirectoryError: raised whenever passed folder_path is invalid
        """
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


class ConcurrentPickledDicomObjectTaskSplitter(DicomoDataset):
    """This class server as a proxy for the underlying iterators when the iter() method is called it returns a
    PickledDicomoFilePool object which can be iterated over, this results in a class that supports a parallel data
    loading. It's constructor takes in the following parameters:

    :param base_path: a valid path to a folder containing a .gemset data sets
    :type base_path: str
    :param labels: labels specifying which DataObject values except for a tensor will be returned by the next() call
    :type labels: list
    :param transform: optional transforms to be applied on the tensor
    :type transform: Optional[any torchvision.transforms]
    :param constraints: optional constraints that the DataObject has to fulfil in order to be returned by the
        next() call, eg. {'Modality': 'CT'} or {'Modality': ['CT', 'MG']}
    :type constraints: Optional[dict]
    :param label_counter_type: label counter used by the summarize method
    :type label_counter_type: gemicai.label_counter.GemicaiLabelCounter
    :raises TypeError: raised if any of the parameters has an invalid type
    """

    def __init__(self, base_path, labels, transform=None, constraints={},
                 label_counter_type=gem.label_counters.LabelCounter):
        if not isinstance(labels, list):
            raise TypeError('labels is not a list')
        if not isinstance(base_path, str):
            raise TypeError('base_path is not a string')
        if not isinstance(constraints, dict):
            raise TypeError('constraints is not a dict')
        super(ConcurrentPickledDicomObjectTaskSplitter, self).__init__(label_counter_type)

        self.labels = labels
        self.constraints = constraints
        self.base_path = base_path
        self.transform = transform
        self.len = 0

        self.file_pool = []
        for root, dirs, files in os.walk(self.base_path):
            dirs.sort()
            for name in sorted(files):
                self.file_pool.append(os.path.join(root, name))

    def __iter__(self):
        """If there are worker threads present it divides a file pool between them otherwise pool is left untouched and
        the PickledDicomoFilePool object is returned.

        :return: a valid gemicai.data_iterators.PickledDicomoFilePool object
        """
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
        """:raises Exception: Not supported by this class"""
        raise Exception("This 'Iterator' is meant to split a file pool and return PickledDicomoFilePool")

    def __len__(self):
        """:raises Exception: Not supported by this class"""
        raise Exception("This 'Iterator' is meant to split a file pool and return PickledDicomoFilePool")

    def can_be_parallelized(self):
        """This iterator supports a parallelized resource loading.

        :return: always returns True
        """
        return True

    def subset(self, constraints):
        """Returns a data set subset using provided constraints. Subset is created by merging current data set
        constraints with the ones passed to this method. In order to store only the DataObjects fulfilling those
        constraints please refer to the save method.

        :param constraints: optional constraints that the DataObject has to fulfil in order to be returned by the
            next() call, eg. {'Modality': 'CT'} or {'Modality': ['CT', 'MG']}
        :type constraints: dict
        :return: a valid gemicai.data_iterators.ConcurrentPickledDicomObjectTaskSplitter object
        :raises TypeError: raised whenever constraints parameter is not a dict
        """
        if not isinstance(constraints, dict):
            raise TypeError('constraints is not a dict')
        return ConcurrentPickledDicomObjectTaskSplitter(self.base_path, self.labels, self.transform,
                                                        {**self.constraints, **constraints})

    def summarize(self, label, print_summary=True):
        dataset = self.__iter__()

        if not print_summary:
            return dataset.summarize(label, print_summary)
        dataset.summarize(label, print_summary)

    def plot_one_of_every(self, label, cmap='gray_r'):
        return self.__iter__().plot_one_of_every(label, cmap)

    def save(self, directory):
        """Saves DataObjects fulfilling the iterator's constraints to the specified directory as a .gemset files.

        :param directory: a valid directory path to which data sets will be saved
        :type directory: str
        """
        PickledDicomoFilePool(self.file_pool, self.labels, self.transform, self.constraints).save(directory)

    def erase(self):
        """Erases iterator's data sets from the file system."""
        PickledDicomoFilePool(self.file_pool, self.labels, self.transform, self.constraints).erase()

    def modify(self, index, fields):
        """Modifies the underlying DataObject with a given index using a provided dictionary with a field-value
        mappings, eg. set.modify(2, {'Modality': 'CT'}) sets 'Modality' of a third object in the data set to 'CT'.
        Note that the first object in the data set has an index equal to zero.

        :param index: a non-negative index of the object to modify
        :type  index: int
        :param fields: fields to modify, eg. {'Modality': 'CT', ...}
        :type  fields: dict
        """
        PickledDicomoFilePool(self.file_pool, self.labels, self.transform, self.constraints).modify(index, fields)

    def split(self, sets={'train': 0.8, 'test': 0.2}, max_objects_per_file=1000, self_erase_afterwards=False):
        """Splits the underlying original data set into N-sets specified by the sets parameter. Each data set that
        is a file with a .gemset extension should contain up to a max_objects_per_file DataObjects as a result. If
        self_erase_afterwards is set to True the original data set will be erased.

        :param sets: dictionary with a valid path-ratio mappings, eg.
            sets={'train': 0.8, 'test': 0.2} should split the original data set into two sets with a ratio of 8:2. First
            set should be saved into a 'train' folder and the second into a 'test' folder. The sum of all the ratios
            added up all together should be equal to 1.0.
        :type sets: dict
        :param max_objects_per_file: a non-negative integer specifying how many objects at maximum can be inside of each
            .gemset file
        :type max_objects_per_file: int
        :param self_erase_afterwards: specifies whenever the original data set should be removed after splitting or not.
        :type self_erase_afterwards: bool
        """
        PickledDicomoFilePool(self.file_pool, self.labels, self.transform, self.constraints).\
            split(sets, max_objects_per_file, self_erase_afterwards)


class PickledDicomoFilePool(DicomoDataset):
    """This class takes in a list of files as an input and iterates over them.
    It's constructor takes in the following parameters:

    :param file_pool: list of a valid file paths to .gemset data sets
    :type file_pool: list
    :param labels: labels specifying which DataObject values except for a tensor will be returned by the next() call
    :type labels: list
    :param transform: optional transforms to be applied on the tensor
    :type transform: Optional[any torchvision.transforms]
    :param constraints: optional constraints that the DataObject has to fulfil in order to be returned by the
        next() call, eg. {'Modality': 'CT'} or {'Modality': ['CT', 'MG']}
    :type constraints: Optional[dict]
    :param label_counter_type: label counter used by the summarize method
    :type label_counter_type: gemicai.label_counter.GemicaiLabelCounter
    :raises TypeError: raised if any of the parameters has an invalid type
    :raises FileNotFoundError: raised if some path in the file_pool does not point to any existing file
    """

    def __init__(self, file_pool, labels, transform=None, constraints={},
                 label_counter_type=gem.label_counters.LabelCounter):
        if not isinstance(labels, list):
            raise TypeError('dicomo_fields is not a list')
        if not isinstance(file_pool, list):
            raise TypeError('file_pool is not a list')
        for path in file_pool:
            if not os.path.isfile(path):
                raise FileNotFoundError(path + " does not point to any existing file")
        if not isinstance(constraints, dict):
            raise TypeError('constraints is not a dict')
        super(PickledDicomoFilePool, self).__init__(label_counter_type)

        self.labels = labels
        self.constraints = constraints
        self.file_pool = file_pool
        self.transform = transform
        self.set_generator = None
        self.data_set = None
        self.len = 0

    def __iter__(self):
        """Prepares data iterator such that next() can be called on it

        :return: self
        """
        self.set_generator = self._pool_walker()
        self.data_set = next(self.set_generator)
        return self

    def __next__(self):
        """Returns list containing a tensor and selected label values

        :return: list containing a tensor and selected label values
        :raises StopIteration: raised when there is no more data left to iterate over
        :raises Exception: raised when the specified transformation cannot be applied to the tensor
        """
        while True:
            try:
                temp = next(self.data_set)
                self.len += 1
                return temp
            except StopIteration:
                self.data_set = next(self.set_generator)

    def __len__(self):
        """:return: number of objects iterated over so far"""
        return self.len

    def can_be_parallelized(self):
        """This iterator does not support a parallelized resource loading.

        :return: always returns False
        """
        return False

    def _pool_walker(self):
        """used internally in order to fetch a next PickledDicomoDataSet"""
        for file_path in self.file_pool:
            yield iter(PickledDicomoDataSet(file_path, self.labels, self.transform, self.constraints))

    def subset(self, constraints):
        """Returns a data set subset using provided constraints. Subset is created by merging current data set
        constraints with the ones passed to this method. In order to store only the DataObjects fulfilling those
        constraints please refer to the save method.

        :param constraints: optional constraints that the DataObject has to fulfil in order to be returned by the
            next() call, eg. {'Modality': 'CT'} or {'Modality': ['CT', 'MG']}
        :type constraints: dict
        :return: a valid gemicai.data_iterators.PickledDicomoFilePool object
        :raises TypeError: raised whenever constraints parameter is not a dict
        """
        if not isinstance(constraints, dict):
            raise TypeError('constraints is not a dict')
        return PickledDicomoFilePool(self.file_pool, self.labels, self.transform, {**self.constraints, **constraints})

    def save(self, directory):
        """Saves DataObjects fulfilling the iterator's constraints to the specified directory as a .gemset files.

        :param directory: a valid directory path to which data sets will be saved
        :type directory: str
        """
        for file_path in self.file_pool:
            PickledDicomoDataSet(file_path, self.labels, self.transform, self.constraints).save(directory)

    def erase(self):
        """Erases iterator's data sets from the file system."""
        for file_path in self.file_pool:
            PickledDicomoDataSet(file_path, self.labels, self.transform, self.constraints).erase()

    def modify(self, index, fields):
        """Modifies the underlying DataObject with a given index using a provided dictionary with a field-value
        mappings, eg. set.modify(2, {'Modality': 'CT'}) sets 'Modality' of a third object in the data set to 'CT'.
        Note that the first object in the data set has an index equal to zero.

        :param index: a non-negative index of the object to modify
        :type  index: int
        :param fields: fields to modify, eg. {'Modality': 'CT', ...}
        :type  fields: dict
        """
        if not isinstance(index, int) or index < 0:
            raise TypeError("index should be a non-negative int")
        if not isinstance(fields, dict):
            raise TypeError("fields should be a dict, eg. {'Modality': 'CT', ...} here a value of the Modality label "
                            "will be changed to CT")

        # since we have to preserve this iterator's state let's create a new one
        dataset = iter(PickledDicomoFilePool(self.file_pool, self.labels, self.transform, self.constraints))

        try:
            # forward internal pointer to the correct PickledDicomoDataSet
            while 0 <= index:
                next(dataset)
                index -= 1

            # modify underlying dataset
            dataset.data_set.modify(len(dataset.data_set) - 1, fields)

        except StopIteration:
            # turns out that the size of index is bigger (or equal) than the count of actual objects in the data set
            raise IndexError("given index is out of bounds for the given dataset")

    def split(self, sets={'train': 0.8, 'test': 0.2}, max_objects_per_file=1000, self_erase_afterwards=False):
        """Splits the underlying original data set into N-sets specified by the sets parameter. Each data set that
        is a file with a .gemset extension should contain up to a max_objects_per_file DataObjects as a result. If
        self_erase_afterwards is set to True the original data set will be erased.

        :param sets: dictionary with a valid path-ratio mappings, eg.
            sets={'train': 0.8, 'test': 0.2} should split the original data set into two sets with a ratio of 8:2. First
            set should be saved into a 'train' folder and the second into a 'test' folder. The sum of all the ratios
            added up all together should be equal to 1.0.
        :type sets: dict
        :param max_objects_per_file: a non-negative integer specifying how many objects at maximum can be inside of each
            .gemset file
        :type max_objects_per_file: int
        :param self_erase_afterwards: specifies whenever the original data set should be removed after splitting or not.
        :type self_erase_afterwards: bool
        """
        if not isinstance(sets, dict):
            raise TypeError("sets parameter should be a dict and have a format {'file_path_1': ratio_1, "
                            "'file_path_2: ratio_2'}, note that sum of ratios should add up to a 1")
        if not isinstance(max_objects_per_file, int) or max_objects_per_file <= 0:
            raise TypeError("max_objects_per_file parameter should be a positive int")
        if not isinstance(self_erase_afterwards, bool):
            raise TypeError("self_erase_afterwards parameter should be a bool")

        ratio_sum = 0.0
        for path in sets:
            if not os.path.isdir(path):
                raise ValueError(path + " is not a valid folder path")
            ratio_sum += sets[path]

        if ratio_sum != 1.0:
            raise ValueError("ratios added up all together should result in 1.0")

        # TODO: it would be smart to hold information how many objects there are in a dataset to not count that
        #  everytime
        # TODO all of this has to be done because of how pickle works, maybe we should exchange it or implement our
        #  own file format?

        # find out how many items there are in this data set
        # since we have to preserve this iterator's state let's create a new one
        dataset = iter(PickledDicomoFilePool(self.file_pool, self.labels, self.transform, self.constraints))
        objects = 0

        try:
            while True:
                data_object = next(dataset)
                objects += 1
        except StopIteration:
            None

        # calculate how many files each set should hold in the end
        should_get = [round(objects * sets[key]) for key in sets]

        # sanity check
        sum = 0
        for file_count in should_get:
            sum += file_count
        if sum != objects:
            raise AssertionError("it is impossible to split given data set using current ratio")

        # objects left before a new temp file has to be created, per set basis
        objects_left = [max_objects_per_file for _ in range(len(sets))]

        # generators for the file names, each set gets its own
        folder_paths = [directory for directory in sets]
        file_names = [("%06i.gemset" % i for i in count(1)) for directory in sets]

        # create a temp file per set
        temp_files = [gem.tempfile.NamedTemporaryFile(mode="ab+", delete=False) for _ in range(len(sets))]
        try:
            # reset iterator's internal pointer
            dataset = iter(dataset)

            current_file = 0
            file_number = len(temp_files)

            # small function for fetching a new data object
            def next_object(dataset):
                while True:
                    try:
                        return next(dataset.data_set.pickle_stream)
                    except EOFError:
                        dataset.data_set = next(dataset.set_generator)
            obj = next_object(dataset)

            # split dataset
            while True:
                if should_get[current_file]:

                    # fetch next data object and try to dump it into a temp file
                    gemicai.io.pickle.dump(obj, temp_files[current_file])
                    objects_left[current_file] -= 1
                    should_get[current_file] -= 1

                    # we have reached max_objects_per_file limit
                    if not objects_left[current_file]:
                        temp = temp_files[current_file]
                        temp.flush()

                        # write the temp file to the file system
                        gemicai.io.pickle.zip_to_file(temp, os.path.join(folder_paths[current_file],
                                                                         next(file_names[current_file])))
                        objects_left[current_file] = max_objects_per_file

                        # clear the temp file's content
                        temp.seek(0)
                        temp.truncate()

                    obj = next_object(dataset)

                # advance to the next file
                current_file = (current_file + 1) % file_number

        except StopIteration:
            # no more objects left, save temp file's content
            for index, path in enumerate(sets):
                # if temp file its not empty copy it's content
                if objects_left[index] != max_objects_per_file:
                    temp_files[index].flush()
                    gemicai.io.pickle.zip_to_file(temp_files[index], os.path.join(folder_paths[index],
                                                                                  next(file_names[index])))

        finally:
            # now remove created temp files
            for temp in temp_files:
                temp.close()
                os.remove(temp.name)

        if self_erase_afterwards:
            self.erase()


# TODO whole PickledDicomoDataFolder could be implemented in terms of PickledDicomoFilePool at this point
class PickledDicomoDataFolder(DicomoDataset):
    """This class takes in a path to a folder containing a .gemset data sets and iterates over them.
    It's constructor takes in the following parameters:

    :param base_path: a path to a valid folder containing a .gemset data sets
    :type base_path: str
    :param labels: labels specifying which DataObject values except for a tensor will be returned by the next() call
    :type labels: list
    :param transform: optional transforms to be applied on the tensor
    :type transform: Optional[any torchvision.transforms]
    :param constraints: optional constraints that the DataObject has to fulfil in order to be returned by the
        next() call, eg. {'Modality': 'CT'} or {'Modality': ['CT', 'MG']}
    :type constraints: Optional[dict]
    :param label_counter_type: label counter used by the summarize method
    :type label_counter_type: gemicai.label_counter.GemicaiLabelCounter
    :raises TypeError: raised if any of the parameters has an invalid type
    :raises NotADirectoryError: raised if the passed path does not point to any directory
    """

    def __init__(self, base_path, labels, transform=None, constraints={},
                 label_counter_type=gem.label_counters.LabelCounter):
        if not isinstance(labels, list):
            raise TypeError('labels is not a list')
        if not isinstance(base_path, str):
            raise TypeError('base_path is not a string')
        if not os.path.isdir(base_path):
            raise NotADirectoryError("base_path does not point to any existing directory")
        if not isinstance(constraints, dict):
            raise TypeError('constraints is not a dict')
        super(PickledDicomoDataFolder, self).__init__(label_counter_type)

        self.labels = labels
        self.base_path = base_path
        self.transform = transform
        self.constraints = constraints
        self.len = 0

    def __iter__(self):
        """Prepares data iterator such that next() can be called on it

       :return: self
       """
        self.data_set_gen = self._get_next_data_set()
        self.data_set = next(self.data_set_gen)
        self.len = 0
        return self

    def __next__(self):
        """Returns list containing a tensor and selected label values

        :return: list containing a tensor and selected label values
        :raises StopIteration: raised when there is no more data left to iterate over
        :raises Exception: raised when the specified transformation cannot be applied to the tensor
        """
        while True:
            try:
                temp = next(self.data_set)
                self.len += 1
                return temp
            except StopIteration:
                self.data_set = next(self.data_set_gen)

    def __len__(self):
        """:return: number of objects iterated over so far"""
        return self.len

    def _get_next_data_set(self):
        """used internally in order to fetch a next PickledDicomoDataSet"""
        for root, dirs, files in os.walk(self.base_path):
            dirs.sort()
            for name in sorted(files):
                yield iter(PickledDicomoDataSet(os.path.join(root, name), self.labels, self.transform,
                                                self.constraints))

    def can_be_parallelized(self):
        """This iterator does not support a parallelized resource loading.

        :return: always returns False
        """
        return False

    def subset(self, constraints):
        """Returns a data set subset using provided constraints. Subset is created by merging current data set
         constraints with the ones passed to this method. In order to store only the DataObjects fulfilling those
         constraints please refer to the save method.

        :param constraints: optional constraints that the DataObject has to fulfil in order to be returned by the
            next() call, eg. {'Modality': 'CT'} or {'Modality': ['CT', 'MG']}
        :type constraints: dict
        :return: a valid gemicai.data_iterators.PickledDicomoDataFolder object
        :raises TypeError: raised whenever constraints parameter is not a dict
        """
        if not isinstance(constraints, dict):
            raise TypeError('constraints is not a dict')
        return PickledDicomoDataFolder(self.base_path, self.labels, self.transform, {**self.constraints, **constraints})

    def _get_underlying_file_pool(self):
        file_pool = []
        for root, dirs, files in os.walk(self.base_path):
            dirs.sort()
            for name in sorted(files):
                file_pool += [os.path.join(root, name)]
        return file_pool

    def save(self, directory):
        """Saves DataObjects fulfilling the iterator's constraints to the specified directory as a .gemset files.

         :param directory: a valid directory path to which data sets will be saved
         :type directory: str
         """
        PickledDicomoFilePool(self._get_underlying_file_pool(), self.labels, self.transform, self.constraints).\
            save(directory)

    def erase(self):
        """Erases iterator's data sets from the file system."""
        PickledDicomoFilePool(self._get_underlying_file_pool(), self.labels, self.transform, self.constraints).erase()

    def modify(self, index, fields):
        """Modifies the underlying DataObject with a given index using a provided dictionary with a field-value
        mappings, eg. set.modify(2, {'Modality': 'CT'}) sets 'Modality' of a third object in the data set to 'CT'.
        Note that the first object in the data set has an index equal to zero.

        :param index: a non-negative index of the object to modify
        :type  index: int
        :param fields: fields to modify, eg. {'Modality': 'CT', ...}
        :type  fields: dict
        """
        PickledDicomoFilePool(self._get_underlying_file_pool(), self.labels, self.transform, self.constraints).\
            modify(index, fields)

    def split(self, sets={'train': 0.8, 'test': 0.2},  max_objects_per_file=1000, self_erase_afterwards=False):
        """Splits the underlying original data set into N-sets specified by the sets parameter. Each data set that
        is a file with a .gemset extension should contain up to a max_objects_per_file DataObjects as a result. If
        self_erase_afterwards is set to True the original data set will be erased.

        :param sets: dictionary with a valid path-ratio mappings, eg.
            sets={'train': 0.8, 'test': 0.2} should split the original data set into two sets with a ratio of 8:2. First
            set should be saved into a 'train' folder and the second into a 'test' folder. The sum of all the ratios
            added up all together should be equal to 1.0.
        :type sets: dict
        :param max_objects_per_file: a non-negative integer specifying how many objects at maximum can be inside of each
            .gemset file
        :type max_objects_per_file: int
        :param self_erase_afterwards: specifies whenever the original data set should be removed after splitting or not.
        :type self_erase_afterwards: bool
        """
        PickledDicomoFilePool(self._get_underlying_file_pool(), self.labels, self.transform, self.constraints).\
            split(sets,  max_objects_per_file, self_erase_afterwards)


class PickledDicomoDataSet(DicomoDataset):
    """This class takes in a valid path to a .gemset data set and iterates over it.
    It's constructor takes in the following parameters:

    :param pickle_path: a path to a valid .gemset file
    :type pickle_path: str
    :param labels: labels specifying which DataObject values except for a tensor will be returned by the next() call
    :type labels: Optional[list]
    :param transform: optional transforms to be applied on the tensor
    :type transform: Optional[any torchvision.transforms]
    :param constraints: optional constraints that the DataObject has to fulfil in order to be returned by the
        next() call, eg. {'Modality': 'CT'} or {'Modality': ['CT', 'MG']}
    :type constraints: Optional[dict]
    :param label_counter_type: label counter used by the summarize method
    :type label_counter_type: gemicai.label_counter.GemicaiLabelCounter
    :raises TypeError: raised if any of the parameters has an invalid type
    :raises FileNotFoundError: raised whenever passed pickle_path does not point to any existing file
    """

    def __init__(self, pickle_path, labels=[], transform=None, constraints={},
                 label_counter_type=gem.label_counters.LabelCounter):
        self.tmp = None

        if not isinstance(labels, list):
            raise TypeError('labels is not a list')
        if not isinstance(pickle_path, str):
            raise TypeError('pickle_path is not a string')
        if not os.path.isfile(pickle_path):
            raise FileNotFoundError("pickle_path does not point to any existing file")
        if not isinstance(constraints, dict):
            raise TypeError('constraints is not a dict')
        super(PickledDicomoDataSet, self).__init__(label_counter_type)

        self.labels = labels
        self.pickle_path = pickle_path
        self.transform = transform
        self.constraints = constraints
        self.len = 0

    def __iter__(self):
        """Prepares data iterator such that next() can be called on it

        :return: self
        """
        self.len = 0
        self.pickle_stream = self._stream_pickled_dicomos()
        return self

    def __next__(self):
        """Returns list containing a tensor and selected label values

        :return: list containing a tensor and selected label values
        :raises StopIteration: raised when there is no more data left to iterate over
        :raises Exception: raised when the specified transformation cannot be applied to the tensor
        :raises TypeError: raised whenever a file pointed by the pickle_path does not contain a valid
            gemicai.data_objects.DicomObject object
        """
        try:
            try:
                # get next dicomo class from the stream
                dicomo_class = next(self.pickle_stream)
                if not isinstance(dicomo_class, gemicai.data_objects.DicomObject):
                    raise TypeError("pickled data set should contain gemicai.data_iterators.DicomObject but it contains "
                                    + type(dicomo_class))

                if not self._meets_constraints(dicomo_class):
                    return self.__next__()

                if len(self.labels) == 0:
                    return dicomo_class

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
                    labels.append(dicomo_class.get(label))

                self.len += 1

                # All hail to python for not returning reference to the temp object after calling .append
                # which forces me to do this
                return [tensor] + labels
            except:
                self._file_cleanup()
                raise
        except EOFError:
            raise StopIteration

    def __len__(self):
        """:return: number of objects iterated over so far"""
        return self.len

    def __del__(self):
        """Closes a file when it goes out of scope"""
        self._file_cleanup()

    def _file_cleanup(self):
        """Closes an open file"""
        if self.tmp is not None:
            self.tmp.close()
            os.remove(self.tmp.name)
            self.tmp = None

    def _stream_pickled_dicomos(self):
        """used internally in order to fetch a next instance of gemicai.data_objects.DataObject"""
        self.tmp = gem.tempfile.NamedTemporaryFile(mode="ab+", delete=False)
        try:
            gem.io.unzip_to_file(self.tmp, self.pickle_path)
            while True:
                yield gem.pickle.load(self.tmp)
        finally:
            pass

    def can_be_parallelized(self):
        """This iterator does not support a parallelized resource loading.

        :return: always returns False
        """
        return False

    def subset(self, constraints):
        """Returns a data set subset using provided constraints. Subset is created by merging current data set
        constraints with the ones passed to this method. In order to store only the DataObjects fulfilling those
        constraints please refer to the save method.

        :param constraints: optional constraints that the DataObject has to fulfil in order to be returned by the
            next() call, eg. {'Modality': 'CT'} or {'Modality': ['CT', 'MG']}
        :type constraints: dict
        :return: a valid gemicai.data_iterators.PickledDicomoDataSet object
        :raises TypeError: raised whenever constraints parameter is not a dict
        """

        if not isinstance(constraints, dict):
            raise TypeError('constraints is not a dict')
        return PickledDicomoDataSet(self.pickle_path, self.labels, self.transform, {**self.constraints, **constraints})

    def _meets_constraints(self, dicomo_class):
        # constraints is a dictionary.
        for k, v in self.constraints.items():
            if isinstance(v, list):
                if dicomo_class.get(k) not in v:
                    return False

            else:
                if dicomo_class.get(k) != v:
                    return False
        return True

    def save(self, directory):
        """Saves DataObjects fulfilling the iterator's constraints to the specified directory as a .gemset file.

        :param directory: a valid directory path to which data sets will be saved
        :type directory: str
        """
        self._file_cleanup()

        if not os.path.isdir(directory):
            raise NotADirectoryError

        # open temp file
        temp = gem.tempfile.NamedTemporaryFile(mode="ab+", delete=False)
        empty = True
        try:
            # fetch next DataObjects until there are none left
            try:
                for dicomo_class in self._stream_pickled_dicomos():
                    # if a DataObject meets the data set constraints put them into temp file
                    if self._meets_constraints(dicomo_class):
                        gemicai.io.pickle.dump(dicomo_class, temp)
                        empty = False
            except EOFError:
                None
            if not empty:
                # if the data set is not empty zip temp file's data to a specified folder
                temp.flush()
                gemicai.io.pickle.zip_to_file(temp, os.path.join(directory, os.path.basename(self.pickle_path)))
        finally:
            temp.close()
            os.remove(temp.name)

    def modify(self, index, fields):
        """Modifies the underlying DataObject with a given index using a provided dictionary with a field-value
        mappings, eg. set.modify(2, {'Modality': 'CT'}) sets 'Modality' of a third object in the data set to 'CT'.
        Note that the first object in the data set has an index equal to zero.

        :param index: a non-negative index of the object to modify
        :type  index: int
        :param fields: fields to modify, eg. {'Modality': 'CT', ...}
        :type  fields: dict
        """
        if not isinstance(index, int) or index < 0:
            raise TypeError("index should be a non-negative int")
        if not isinstance(fields, dict):
            raise TypeError("fields should be a dict, eg. {'Modality': 'CT', ...} here a value of the Modality label "
                            "will be changed to CT")

        # since we have to preserve this iterators state let's create a new one
        dataset = iter(PickledDicomoDataSet(self.pickle_path, self.labels, self.transform, self.constraints))
        temp = gem.tempfile.NamedTemporaryFile(mode="ab+", delete=False)
        try:
            # TODO all of this has to be done because of how pickle works, maybe we should exchange it or implement our
            #  own file format?

            # advance internal data pointer, copy all of the files inside to a new temp file
            while index:
                data_object = next(dataset.pickle_stream)
                gemicai.io.pickle.dump(data_object, temp)
                index -= 1

            # fetch an object to modify
            data_object = next(dataset.pickle_stream)

            # modify it
            for key in fields:
                data_object.set(key, fields[key])

            # write it back
            gemicai.io.pickle.dump(data_object, temp)

            # copy rest of the remaining files in the dataset
            try:
                while True:
                    data_object = next(dataset.pickle_stream)
                    gemicai.io.pickle.dump(data_object, temp)
            except EOFError:
                None

            # exchange the data set
            temp.flush()
            gemicai.io.pickle.zip_to_file(temp, self.pickle_path)

        except EOFError:
            # turns out that the size of index is bigger (or equal) than the count of actual objects in the data set
            raise IndexError("given index is out of bounds for the given data set")
        finally:
            temp.close()
            os.remove(temp.name)

    def erase(self):
        """Erases iterator's data set from the file system."""
        self._file_cleanup()
        os.remove(self.pickle_path)

    def split(self, sets={'train': 0.8, 'test': 0.2}, max_objects_per_file=1000, self_erase_afterwards=False):
        """Splits the underlying original data set into N-sets specified by the sets parameter. Each data set that
         is a file with a .gemset extension should contain up to a max_objects_per_file DataObjects as a result. If
         self_erase_afterwards is set to True the original data set will be erased.

         :param sets: dictionary with a valid path-ratio mappings, eg.
            sets={'train': 0.8, 'test': 0.2} should split the original data set into two sets with a ratio of 8:2.
            First set should be saved into a 'train' folder and the second into a 'test' folder. The sum of all the
            ratios added up all together should be equal to 1.0.
         :type sets: dict
         :param max_objects_per_file: a non-negative integer specifying how many objects at maximum can be inside of
            each .gemset file
         :type max_objects_per_file: int
         :param self_erase_afterwards: specifies whenever the original data set should be removed after splitting or
            not.
         :type self_erase_afterwards: bool
         """
        if not isinstance(sets, dict):
            raise TypeError("sets parameter should be a dict and have a format {'file_path_1': ratio_1, "
                            "'file_path_2: ratio_2'}, note that sum of ratios should add up to a 1")
        if not isinstance(max_objects_per_file, int) or max_objects_per_file <= 0:
            raise TypeError("max_objects_per_file parameter should be a positive int")
        if not isinstance(self_erase_afterwards, bool):
            raise TypeError("self_erase_afterwards parameter should be a bool")

        ratio_sum = 0.0
        for path in sets:
            if not os.path.isdir(path):
                raise ValueError(path + " is not a valid folder path")
            ratio_sum += sets[path]

        if ratio_sum != 1.0:
            raise ValueError("ratios added up all together should result in 1.0")

        # TODO: it would be smart to hold information how many objects there are in a data set to not count that
        #  everytime
        # TODO all of this has to be done because of how pickle works, maybe we should exchange it or implement our
        #  own file format?

        # find out how many items there are in this data set
        # since we have to preserve this iterators state let's create a new one
        dataset = iter(PickledDicomoDataSet(self.pickle_path, self.labels, self.transform, self.constraints))
        objects = 0

        try:
            while True:
                data_object = next(dataset.pickle_stream)
                objects += 1
        except EOFError:
            None

        # calculate how many files each set should hold in the end
        should_get = [round(objects * sets[key]) for key in sets]

        # sanity check
        sum = 0
        for file_count in should_get:
            sum += file_count
        if sum != objects:
            raise AssertionError("it is impossible to split given data set using current ratio")

        # objects left before a new temp file has to be created, per set basis
        objects_left = [max_objects_per_file for _ in range(len(sets))]

        # generators for the file names, each set gets its own
        folder_paths = [directory for directory in sets]
        file_names = [("%06i.gemset" % i for i in count(1)) for directory in sets]

        # create a temp file per set
        temp_files = [gem.tempfile.NamedTemporaryFile(mode="ab+", delete=False) for _ in range(len(sets))]
        try:
            # reset iterator's internal pointer
            dataset = iter(dataset)

            current_file = 0
            file_number = len(temp_files)
            obj = next(dataset.pickle_stream)

            # split data set
            while True:
                if should_get[current_file]:

                    # fetch next data object and try to dump it into a temp file
                    gemicai.io.pickle.dump(obj, temp_files[current_file])
                    objects_left[current_file] -= 1
                    should_get[current_file] -= 1

                    # we have reached max_objects_per_file limit
                    if not objects_left[current_file]:
                        temp = temp_files[current_file]
                        temp.flush()

                        # write the temp file to the file system
                        gemicai.io.pickle.zip_to_file(temp, os.path.join(folder_paths[current_file],
                                                                         next(file_names[current_file])))
                        objects_left[current_file] = max_objects_per_file

                        # clear the temp file's content
                        temp.seek(0)
                        temp.truncate()

                    obj = next(dataset.pickle_stream)

                # advance to the next file
                current_file = (current_file + 1) % file_number

        except EOFError:
            # no more objects left, save temp file's content
            for index, path in enumerate(sets):
                # if temp file its not empty copy it's content
                if objects_left[index] != max_objects_per_file:
                    temp_files[index].flush()
                    gemicai.io.pickle.zip_to_file(temp_files[index], os.path.join(folder_paths[index],
                                                                                  next(file_names[index])))

        finally:
            # now remove created temp files
            for temp in temp_files:
                temp.close()
                os.remove(temp.name)

        if self_erase_afterwards:
            self.erase()
