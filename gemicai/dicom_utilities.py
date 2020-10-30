"""This module contains some utility functions that are used by the Gemicai in order to interface with dicom objects"""

from matplotlib import pyplot as plt
from datetime import datetime
from itertools import count
import gemicai.data_objects
import pydicom as dicom
import torchvision
import tempfile
import gemicai
import pickle
import torch
import numpy
import gzip
import math
import os


def load_dicom(filename):
    """Loads in a given dicom file using a pydicom library

    :param filename: a path to the .dcm.gz or .dcm file
    :type filename: Union[str, os.path]
    :return: pydicom.dataset.FileDataset or pydicom.dicomdir.DicomDir
    :raises TypeError: raised if the file extension does not end with .dcm nor .gz
    """
    if filename.endswith('.dcm'):
        ds = dicom.dcmread(filename)
    elif filename.endswith('.gz'):
        with gzip.open(filename) as fd:
            ds = dicom.dcmread(fd, force=True)
    else:
        raise TypeError
    ds.file_meta.TransferSyntaxUID = dicom.uid.ImplicitVRLittleEndian
    return ds


def plot_dicom_file(dcm, cmap='gray'):
    """Plots image stored in a given dicom file. If a path given instead it will try to load a specified file first.

    :param dcm: dicom object or a valid path to a dicom file
    :type dcm: Union[str, pydicom.dataset.FileDataset]
    :param cmap: color scheme
    :type cmap: str
    """
    if isinstance(dcm, str):
        dcm = load_dicom(dcm)
    plt.imshow(extract_tensor(dcm)[0], cmap)
    plt.show()


def extract_tensor(ds: dicom.Dataset):
    """Extracts an image from the dicom file and creates a tensor out of it

    :param ds: dicom object to extract an image from
    :type ds: pydicom.dataset.FileDataset
    :return: torch.Tensor
    """
    # try to load a dicom file
    # transform pixel_array into a format accepted by the pytorch
    norm = plt.Normalize(vmin=ds.pixel_array.min(), vmax=ds.pixel_array.max())
    data = torch.from_numpy(norm(ds.pixel_array).astype(numpy.float32))

    # if we want to print the resulting image remove the last transform and call tensor.show() after create_tensor
    create_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Grayscale(3),
        torchvision.transforms.Resize((244, 244)),
        torchvision.transforms.ToTensor()
    ])

    tensor = create_tensor(data)
    return tensor


def dicom_to_gemset(data_origin, data_destination, relevant_labels, field_values=[], objects_per_file=1000,
                    pick_middle=False, verbosity=0):
    """Creates a Gemicai dataset from the data_origin (it should contain a valid dicom objects) and puts them in the
    data_destination.

    :param data_origin: path to a folder containing dicom files
    :type data_origin: Union[str, io.path]
    :param data_destination: path to a destination where the gemsets will be outputted
    :type data_destination: Union[str, io.path]
    :param relevant_labels: specify which labels along with their values to extract from the dicom file and put into
        gemicai.data_objects.DicomObject, eg. ['Modality'] in this case DicomObject will contain a tensor and its
        Modality
    :type relevant_labels: list
    :param field_values: dataset will contain only objects which fulfil specified critieria,
        eg. [('Modality', ['CT', 'MG']), ...] in this case dataset will contain only objects whose Modality is set to
        CT or MG
    :type field_values: Optional[list]
    :param objects_per_file: specifies how many objects one gemicai dataset should contain. A default value is 1000
    :type objects_per_file: Optional[int]
    :param pick_middle: specifies whenever instead of taking all images from the series only the middle one is taken.
        This can be useful if someone is dealing with series spanning a multiple of dicom objects.
    :type pick_middle: bool
    :param verbosity: optional non-negative parameter, if set to one it will output how long it took to process all of
        the data from data_origin
    :type verbosity: int
    :raises NotADirectoryError: raised if data_origin or data_destination does not point to an existing directory
    :raises TypeError: raised if any of the parameters has a wrong type or its value is out of the accepted bounds
    """
    start = datetime.now()
    if not os.path.isdir(data_origin):
        raise NotADirectoryError
    if not os.path.isdir(data_destination):
        raise NotADirectoryError
    if not isinstance(relevant_labels, list):
        raise TypeError("relevant_labels parameter should be a list of strings with a name of a relevant fields to "
                        "fetch and put in the DicomoObject")
    if not isinstance(field_values, list):
        raise TypeError("field_values parameter should be a list of tuples (field_name, [field_values]). This parameter"
                        " allows for filtering which DicomoObjects should be put in a dataset")
    if not isinstance(objects_per_file, int):
        raise TypeError("objects_per_file parameter should be an integer")
    if not isinstance(pick_middle, bool):
        raise TypeError("pick_middle parameter should be a boolean")
    if not isinstance(verbosity, int) or verbosity < 0:
        raise TypeError("verbosity parameter should be a non-negative integer")

    # because of windows we have to manage temp file ourselves
    temp = tempfile.NamedTemporaryFile(mode="ab+", delete=False)

    try:
        # holds names for the gziped files
        filename_iterator = ("%06i.gemset" % i for i in count(1))
        objects_inside = 0

        if pick_middle:
            relevant_labels += ["InstanceNumber"]

        for root, dirs, files in os.walk(data_origin):
            middle_file = str(math.floor(len(files)/2))

            for file in files:
                try:
                    d = gemicai.DicomObject.from_file(root + '/' + file, relevant_labels, tensor_size=(244, 244))
                    pickle_object = True

                    if pick_middle and str(d.get_value_of("InstanceNumber")) != middle_file:
                        continue

                    # check whenever we filter fields of DicomoObject
                    if len(field_values):
                        for field, values in field_values:

                            value = d.get_value_of(field)
                            if isinstance(value, dicom.multival.MultiValue):
                                if len([x for x in value if x in values]) == len(value):
                                    pickle_object = False
                            else:
                                if not d.get_value_of(field) in values:
                                    pickle_object = False
                    
                    if not pickle_object:
                        continue

                    # check if we are not allowed to append more files
                    if objects_inside >= objects_per_file:
                        # gzip temp file and clear its content
                        temp.flush()
                        gemicai.io.zip_to_file(temp, os.path.join(data_destination, next(filename_iterator)))
                        objects_inside = 0
                        temp.seek(0)
                        temp.truncate()

                    # dump binary data to the temp file
                    pickle.dump(d, temp)
                    objects_inside += 1

                    if pick_middle:
                        break

                except Exception as ex:
                    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                    message = template.format(type(ex).__name__, ex.args)
                    print(message)

        temp.flush()
        gemicai.io.zip_to_file(temp, os.path.join(data_destination, next(filename_iterator)))
    finally:
        temp.close()
        os.remove(temp.name)
    if verbosity >= 1:
        print('Creating .gemset took {}'.format(gemicai.utils.strfdelta(datetime.now() - start)))
