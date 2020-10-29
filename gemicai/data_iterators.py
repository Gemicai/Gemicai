"""This module contains data iterators which are used in order to traverse a dataset and retrieve a relevant information
from the DataObjects it contains."""

from torch.utils.data import get_worker_info
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
import gemicai.data_objects
import gemicai as gem
import math
import os
import matplotlib.pyplot as plt


class GemicaiDataset(ABC, IterableDataset):
    """This interface class serves as a basis for the every Gemicai's data iterator."""

    @abstractmethod
    def __init__(self):
        """Should initialize the iterator, it could for example take in a path to the dataset."""
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
        """Should return or print a summary of all the DataObject values in the dataset selected by the label.

        :param label: field label which values to summarize, for example 'CT' or 'MG'
        :type label: str
        :param print_summary: whenever to print or return an instance of gemicai.label_counters.GemicaiLabelCounter
            object
        :type print_summary: bool
        :return: if print_summary is set to false a class that extends a gemicai.label_counters.GemicaiLabelCounter
        """
        pass

    @abstractmethod
    def subset(self, constraints):
        """Should return a subset of a current dataset.

        :param constraints: dictionary with a dataset constraints, eg. {'CT': 'some_value'}
        :type constraints: dict
        :return: a correct user defined iterator type which extends gemicai.data_iterators.GemicaiDataset
        """
        pass

    @abstractmethod
    def classes(self, label):
        """Should return a list of all the classes in the dataset.

        :param label: label to summarize on
        :type label: str
        :return: list of possible label values present in the dataset
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
        """Should return a boolean specifying whenever current iterator supports parallelized resource loading."""
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

    def classes(self, label):
        """Returns a list of all of the classes in the dataset.

        :param label: label to summarize on
        :type label: str
        :return: list of possible label values present in the dataset
        """
        return list(self.summarize(label, print_summary=False).dic.keys())

    def summarize(self, label, print_summary=True):
        """Returns or prints a summary of all the DataObject values in the dataset selected by the label.

        :param label: field label which values to summarize, for example 'CT' or 'MG'
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
            cnt.update(dicomo.get_value_of(label))
        self.labels = temp
        if print_summary:
            print(cnt)
        else:
            return cnt

    def plot_one_of_every(self, label, cmap='gray_r'):
        """Plots one image per value type.

        :param label: label according to which we will look for a unique values, eg, 'CT'
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
            next() call, eg. {'CT': 'some_value'} or {'CT': ['val_1', 'val_2']}
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

        :param folder_path: a valid path to an existing folder which contains .gemset datasets
        :type folder_path: str
        :param labels: labels specifying which DataObject values except for a tensor will be returned by the next() call
        :type labels: Optional[list]
        :param transform: optional transforms to be applied on the tensor
        :type transform: Optional[any torchvision.transforms]
        :param constraints: optional constraints that the DataObject has to fulfil in order to be returned by the
            next() call, eg. {'CT': 'some_value'} or {'CT': ['val_1', 'val_2']}
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

        :param data_set_path:  a valid path to an existing folder which contains .gemset datasets
            or a valid path to a .gemset file
        :param labels: labels specifying which DataObject values except for a tensor will be returned by the next() call
        :type labels: Optional[list]
        :param constraints: optional constraints that the DataObject has to fulfil in order to be returned by the
            next() call, eg. {'CT': 'some_value'} or {'CT': ['val_1', 'val_2']}
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

    :param base_path: a valid path to a folder containing a .gemset datasets
    :type base_path: str
    :param labels: labels specifying which DataObject values except for a tensor will be returned by the next() call
    :type labels: list
    :param transform: optional transforms to be applied on the tensor
    :type transform: Optional[any torchvision.transforms]
    :param constraints: optional constraints that the DataObject has to fulfil in order to be returned by the
        next() call, eg. {'CT': 'some_value'} or {'CT': ['val_1', 'val_2']}
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
            for name in files:
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
        """Returns a dataset subset using provided constraints

        :param constraints: optional constraints that the DataObject has to fulfil in order to be returned by the
            next() call, eg. {'CT': 'some_value'} or {'CT': ['val_1', 'val_2']}
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


class PickledDicomoFilePool(DicomoDataset):
    """This class takes in a list of files as an input and iterates over them.

    :param file_pool: list of a valid file paths to .gemset datasets
    :type file_pool: list
    :param labels: labels specifying which DataObject values except for a tensor will be returned by the next() call
    :type labels: list
    :param transform: optional transforms to be applied on the tensor
    :type transform: Optional[any torchvision.transforms]
    :param constraints: optional constraints that the DataObject has to fulfil in order to be returned by the
        next() call, eg. {'CT': 'some_value'} or {'CT': ['val_1', 'val_2']}
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
        """Returns a dataset subset using provided constraints

        :param constraints: optional constraints that the DataObject has to fulfil in order to be returned by the
            next() call, eg. {'CT': 'some_value'} or {'CT': ['val_1', 'val_2']}
        :type constraints: dict
        :return: a valid gemicai.data_iterators.PickledDicomoFilePool object
        :raises TypeError: raised whenever constraints parameter is not a dict
        """
        if not isinstance(constraints, dict):
            raise TypeError('constraints is not a dict')
        return PickledDicomoFilePool(self.file_pool, self.labels, self.transform, {**self.constraints, **constraints})


class PickledDicomoDataFolder(DicomoDataset):
    """This class takes in a path to a folder containing a .gemset datasets and iterates over them.

    :param base_path: a path to a valid folder containing a .gemset datasets
    :type base_path: str
    :param labels: labels specifying which DataObject values except for a tensor will be returned by the next() call
    :type labels: list
    :param transform: optional transforms to be applied on the tensor
    :type transform: Optional[any torchvision.transforms]
    :param constraints: optional constraints that the DataObject has to fulfil in order to be returned by the
        next() call, eg. {'CT': 'some_value'} or {'CT': ['val_1', 'val_2']}
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
            for name in files:
                yield iter(PickledDicomoDataSet(os.path.join(root, name), self.labels, self.transform,
                                                self.constraints))

    def can_be_parallelized(self):
        """This iterator does not support a parallelized resource loading.

        :return: always returns False
        """
        return False

    def subset(self, constraints):
        """Returns a dataset subset using provided constraints

        :param constraints: optional constraints that the DataObject has to fulfil in order to be returned by the
            next() call, eg. {'CT': 'some_value'} or {'CT': ['val_1', 'val_2']}
        :type constraints: dict
        :return: a valid gemicai.data_iterators.PickledDicomoDataFolder object
        :raises TypeError: raised whenever constraints parameter is not a dict
        """
        if not isinstance(constraints, dict):
            raise TypeError('constraints is not a dict')
        return PickledDicomoDataFolder(self.base_path, self.labels, self.transform, {**self.constraints, **constraints})


class PickledDicomoDataSet(DicomoDataset):
    """This class takes in a valid path to a .gemset dataset and iterates over it.

    :param pickle_path: a path to a valid .gemset file
    :type pickle_path: str
    :param labels: labels specifying which DataObject values except for a tensor will be returned by the next() call
    :type labels: list
    :param transform: optional transforms to be applied on the tensor
    :type transform: Optional[any torchvision.transforms]
    :param constraints: optional constraints that the DataObject has to fulfil in order to be returned by the
        next() call, eg. {'CT': 'some_value'} or {'CT': ['val_1', 'val_2']}
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
                    raise TypeError("pickled dataset should contain gemicai.data_iterators.DicomObject but it contains "
                                    + type(dicomo_class))

                # constraints is a dictionary.
                meets_constraints = True
                for k, v in self.constraints.items():
                    if isinstance(v, list):
                        if dicomo_class.get_value_of(k) not in v:
                            meets_constraints = False
                            break
                    else:
                        if dicomo_class.get_value_of(k) != v:
                            meets_constraints = False
                            break
                if not meets_constraints:
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
                    labels.append(dicomo_class.get_value_of(label))

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
        """Returns a dataset subset using provided constraints

        :param constraints: optional constraints that the DataObject has to fulfil in order to be returned by the
            next() call, eg. {'CT': 'some_value'} or {'CT': ['val_1', 'val_2']}
        :type constraints: dict
        :return: a valid gemicai.data_iterators.PickledDicomoDataSet object
        :raises TypeError: raised whenever constraints parameter is not a dict
        """
        if not isinstance(constraints, dict):
            raise TypeError('constraints is not a dict')
        return PickledDicomoDataSet(self.pickle_path, self.labels, self.transform, {**self.constraints, **constraints})
