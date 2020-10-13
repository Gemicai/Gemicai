from gemicai.label_counters import LabelCounter
from matplotlib import pyplot as plt
from itertools import count
import gemicai.data_objects
import pydicom as dicom
import torchvision
import tempfile
import shutil
import pickle
import torch
import numpy
import gzip
import os


def load_dicom(filename):
    if filename.endswith('.dcm'):
        ds = dicom.dcmread(filename)
    elif filename.endswith('.gz'):
        with gzip.open(filename) as fd:
            ds = dicom.dcmread(fd, force=True)
    else:
        raise TypeError
    ds.file_meta.TransferSyntaxUID = dicom.uid.ImplicitVRLittleEndian
    return ds


# Returns ({imgage as tensor}, {label})
def dicom_get_tensor_and_label(dicom_file_path):
    # try to load a dicom file
    ds = load_dicom(dicom_file_path)

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
    labels = (getattr(ds, 'BodyPartExamined'), getattr(ds, 'StudyDescription'),
              getattr(ds, 'SeriesDescription'), getattr(ds, 'Modality'))
    # TODO: Figure out direction from which image is taken. e.g. frontal, lateral, top down or down top etc.
    # TODO: this is probably in the dicom header, ask Jeroen about this.
    return tensor, labels


# Plots dicom image with some additional label info.
def plot_dicom(dicom_file_path, cmap=None):
    tensor, label = dicom_get_tensor_and_label(dicom_file_path)
    tensor = tensor.permute(1, 2, 0)
    plt.title('')
    if cmap is None:
        plt.imshow(tensor)
    else:
        plt.imshow(tensor[:, :, 0], cmap=cmap)
    plt.show()


def create_dicomobject_dataset_from_folder(input, output, field_list, field_values=[], objects_per_file=1000):
    if not os.path.isdir(input):
        raise NotADirectoryError
    if not os.path.isdir(output):
        raise NotADirectoryError
    if not isinstance(field_list, list):
        raise TypeError("field_list parameter should be a list of strings with a name of a relevant fields to fetch "
                        "and put in the DicomoObject")
    if not isinstance(field_values, list):
        raise TypeError("field_values parameter should be a list of tuples (field_name, field_values). This parameter "
                        "allows for filtering which DicomoObjects should be put in a dataset")
    if not isinstance(objects_per_file, int):
        raise TypeError("objects_per_file parameter should be an integer")

    # because of windows we have to manage temp file ourselves
    temp = tempfile.NamedTemporaryFile(mode="ab+", delete=False)

    try:
        # counts distinct field values
        cnt = LabelCounter()

        # holds names for the gziped files
        filename_iterator = ("%06i.gemset" % i for i in count(1))
        objects_inside = 0

        for root, dirs, files in os.walk(input):
            for file in files:
                try:
                    d = gemicai.DicomObject.from_file(root + '/' + file, field_list, tensor_size=(244, 244))
                    pickle_object = True

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

                    cnt.update(d.labels)
                    # check if we are not allowed to append more files
                    if objects_inside >= objects_per_file:
                        # gzip temp file and clear its content
                        temp.flush()
                        zip_to_file(temp, os.path.join(output, next(filename_iterator)))
                        objects_inside = 0
                        temp.seek(0)
                        temp.truncate()

                    # dump binary data to the temp file
                    pickle.dump(d, temp)
                    objects_inside += 1

                except Exception as ex:
                    template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                    message = template.format(type(ex).__name__, ex.args)
                    print(message)

        temp.flush()
        zip_to_file(temp, os.path.join(output, next(filename_iterator)))
    finally:
        temp.close()
        os.remove(temp.name)
    return cnt


def zip_to_file(file, zip_path):
    with gzip.open(zip_path, 'wb') as zipped:
        file = open(file.name, 'rb')
        shutil.copyfileobj(file, zipped)
        file.close()


def unzip_to_file(file, zip_path):
    with gzip.open(zip_path, 'rb') as zipped:
        file = open(file.name, 'ab+')
        shutil.copyfileobj(zipped, file)
        file.close()
