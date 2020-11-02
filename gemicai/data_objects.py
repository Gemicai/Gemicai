"""This module contains data objects used by the Gemicai's iterators"""

from gemicai import dicom_utilities as du
from matplotlib import pyplot as plt
from abc import ABC, abstractmethod
import torchvision
import torch
import numpy
import os


class DataObject(ABC):
    """Every custom data object should extend this abstract base class and call it's constructor."""

    def __init__(self, tensor, label_values):
        """Default constructor which enforces some basic rules on it's children.

        :param tensor: contains tensor
        :type tensor: torch.Tensor
        :param label_values: contains a list of tensor's labels
        :type label_values: list
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("DataObject  expects tensor to be a torch.Tensor")
        if not isinstance(label_values, list):
            raise TypeError("DataObject expects labels parameter to be a list")
        self.tensor = tensor
        self.labels = label_values

    @abstractmethod
    def plot(self):
        """Call to this method should plot DataObject's tensor to the screen."""
        pass

    @staticmethod
    @abstractmethod
    def from_file(filename):
        """This method should create and return a DataObject instance.

        :param filename: path to a valid file name
        :type filename: Union[os.path, str]
        """
        pass


class DicomObject(DataObject):
    """Gemicai's default data object """

    def __init__(self, tensor, labels, label_values):
        """Constructs a DicomoObject from the given data.

        :param tensor:
        :type tensor: torch.Tensor
        :param labels: list of image label types
        :type labels: list
        :param label_values: list of image label values
        :type label_values: list
        """
        if not isinstance(labels, list):
            raise TypeError("DataObject expects label_values parameter to be a list")
        DataObject.__init__(self, tensor, label_values)
        self.label_types = labels

    def __str__(self):
        """Returns a string representation of the labels held by this object.

        :return: a string containing a list of tuples in a format (label_type, label_values)
        """
        return str(list(zip(self.label_types, self.labels)))

    # Plots dicom image with some additional label info.
    def plot(self, cmap='gray'):
        """Prints labels and plots the tensor.

        :param cmap: color scheme
        :type cmap: str
        """
        if not isinstance(cmap, str):
            raise TypeError("cmap parameter should be a string")
        plt.title(
            '{}\n{}'.format(self.label_types, self.labels))
        plt.imshow(self.tensor, cmap)
        plt.show()

    def get(self, field):
        """Returns a value of a given field.

        :param field: string with a field label, eg. 'Modality'
        :type field: str
        :return: value of a label or None if the object does not contain it
        """
        if not isinstance(field, str):
            raise TypeError("field parameter should be a string")
        try:
            return self.labels[self.label_types.index(field)]
        except:
            return None

    def set(self, field, value):
        """Sets a specified field to a given value

        :param field: string with a field name, eg. 'Modality'
        :type field: str
        :param value: value that the field will be set to
        :type value: any
        :return True on success, False on failure
        """
        if not isinstance(field, str):
            raise TypeError("field parameter should be a string")

        try:
            self.labels[self.label_types.index(field)] = value
            return True
        except ValueError:
            return False

    def meets_constraints(self, constraints: dict):
        """Checks whenever the object meets a certain type of criteria.

        :param constraints: constraints to check against eg. {'Modality': 'CT'}
        :type constraints: dict
        :return: True if the object meets criteria, False otherwise
        """
        if not isinstance(constraints, dict):
            raise TypeError("constraints parameter should be a dict")
        for k in constraints.keys():
            if self.get(k) != constraints[k]:
                return False
        return True

    @staticmethod
    def from_file(filename, labels, tensor_size=None):
        """Creates a DicomoObject from a specified file.

        :param filename: a valid dicom file path
        :type filename: Union[os.path, str]
        :param labels: labels which values will be taken from the Dicom object. The pixel_array field should not be
            specified as it is one of the default fields fetched internally.
        :type labels: list
        :param tensor_size: used to resize a tensor. If left unspecified it will try to use the current image size
            otherwise it will use the specified values. Correct format ((int)x, (int)y)
        :type tensor_size: Optional[tuple]
        :return: DicomoObject instance
        """
        if not os.path.isfile(filename):
            raise FileNotFoundError
        if not isinstance(labels, list):
            raise TypeError("from_file: fields parameter should be a list of strings but is " +
                            str(type(labels)))
        if not isinstance(tensor_size, tuple) and tensor_size is not None:
            raise TypeError("rom_file: tensor_size parameter should be a tuple of two ints or be set to None but is" +
                            str(type(tensor_size)))

        # try to load a dicom file
        ds = du.load_dicom(filename)

        # transform pixel_array into a format accepted by the pytorch
        norm = plt.Normalize(vmin=ds.pixel_array.min(), vmax=ds.pixel_array.max())
        data = torch.from_numpy(norm(ds.pixel_array).astype(numpy.float32))

        # Because getattr() throws an AttributeError if the field is left empty in the dicom header
        def get_attr(obj, attr):
            try:
                return getattr(obj, attr)
            except AttributeError:
                return None

        if tensor_size is None:
            rows = get_attr(ds, "Rows")
            cols = get_attr(ds, "Columns")
            if rows is None or cols is None:
                raise RuntimeError("Cannot fetch a default tensor sizes, "
                                   "please provide a custom one by passing tensor_size parameter")
            tensor_size = (rows, cols)

        # if we want to print the resulting image remove the last transform and call tensor.show() after create_tensor
        create_tensor = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(tensor_size),
            torchvision.transforms.ToTensor()
        ])

        label_values = []
        # fetch specified labels and return a Dicomo object
        for label in labels:
            label_values.append(get_attr(ds, label))
        return DicomObject(create_tensor(data)[0], labels, label_values)
