from gemicai import dicom_utilities as du
from matplotlib import pyplot as plt
from abc import ABC, abstractmethod
import torchvision
import torch
import numpy
import os


class DataObject(ABC):
    def __init__(self, tensor, label_values):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("DataObject  expects tensor to be a torch.Tensor")
        if not isinstance(label_values, list):
            raise TypeError("DataObject expects labels parameter to be a list")
        self.tensor = tensor
        self.labels = label_values

    @abstractmethod
    def plot(self):
        pass

    @staticmethod
    @abstractmethod
    def from_file(filename):
        pass


class DicomObject(DataObject):
    def __init__(self, tensor, labels, label_values):
        if not isinstance(labels, list):
            raise TypeError("DataObject expects label_values parameter to be a list")
        DataObject.__init__(self, tensor, label_values)
        self.label_types = labels

    def __str__(self):
        return str(list(zip(self.label_types, self.labels)))

    # Plots dicom image with some additional label info.
    def plot(self, cmap='gray'):
        if not isinstance(cmap, str):
            raise TypeError("cmap parameter should be a string")
        plt.title(
            '{}\n{}'.format(self.label_types, self.labels))
        plt.imshow(self.tensor, cmap)
        plt.show()

    def get_value_of(self, item):
        if not isinstance(item, str):
            raise TypeError("item parameter should be a string")
        try:
            return self.labels[self.label_types.index(item)]
        except:
            return None

    def meets_constraints(self, constraints: dict):
        if not isinstance(constraints, dict):
            raise TypeError("constraints parameter should be a dict")
        for k in constraints.keys():
            if self.get_value_of(k) != constraints[k]:
                return False
        return True

    @staticmethod
    def from_file(filename, labels, tensor_size=None):
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
