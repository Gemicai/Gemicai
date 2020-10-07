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
        plt.title(
            '{}\n{}'.format(self.label_types, self.labels))
        plt.imshow(self.tensor, cmap)
        plt.show()

    def get_value_of(self, item):
        try:
            return self.labels[self.label_types.index(item)]
        except:
            return None

    def meets_constraints(self, constraints: dict):
        for k in constraints.keys():
            if self.get_value_of(k) != constraints[k]:
                return False
        return True

    @staticmethod
    def from_file(filename, labels):
        if not os.path.isfile(filename):
            raise FileNotFoundError
        if not isinstance(labels, list):
            raise TypeError("from_file: fields parameter should be a list of strings but is " +
                            str(type(labels)))

        # try to load a dicom file
        ds = du.load_dicom(filename)

        # transform pixel_array into a format accepted by the pytorch
        norm = plt.Normalize(vmin=ds.pixel_array.min(), vmax=ds.pixel_array.max())
        data = torch.from_numpy(norm(ds.pixel_array).astype(numpy.float32))

        # if we want to print the resulting image remove the last transform and call tensor.show() after create_tensor
        create_tensor = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((244, 244)),
            torchvision.transforms.ToTensor()
        ])

        # Because getattr() throws an AttributeError if the field is left empty in the dicom header
        def get_attr(obj, attr):
            try:
                return getattr(obj, attr)
            except AttributeError:
                return None

        label_values = []
        # fetch specified labels and return a Dicomo object
        for label in labels:
            label_values.append(get_attr(ds, label))
        return DicomObject(create_tensor(data)[0], labels, label_values)
