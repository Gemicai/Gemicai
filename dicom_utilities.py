import gzip
import pydicom as dicom
import os
import pandas as pd
from matplotlib import pyplot as plt
import sys
import string
import pathlib
from PIL import Image
import glob
import torchvision
import torch
import numpy

# if needed we can export it later
fields_of_interest = ['Rows', 'StudyDate', 'SeriesTime', 'ContentTime', 'StudyInstanceUID', 'SeriesInstanceUID',
                      'SOPInstanceUID',
                      'Modality', 'SeriesDate', 'AccessionNumber', 'BodyPartExamined', 'StudyDescription',
                      'SeriesDescription', 'InstanceNumber',
                      'PatientOrientation', 'ImageLaterality', 'ImageComments', 'SeriesNumber', 'PatientName']


def load_dicom(filename):
    try:
        if filename.endswith('.dcm'):
            ds = dicom.dcmread(filename)
        else:
            with gzip.open(filename) as fd:
                ds = dicom.dcmread(fd, force=True)
        ds.file_meta.TransferSyntaxUID = dicom.uid.ImplicitVRLittleEndian
    except:
        print('Please priovide a valid .dcm or .dcm.gz file')
        raise
    return ds


def get_os_directory_path(path):
    normalized_path = str(pathlib.Path(path))
    index = normalized_path.rfind('/')
    return os.path.normpath(normalized_path[0:index])


def create_dir_if_does_not_exist(path):
    dir_name = get_os_directory_path(path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def open_directory(path):
    try:
        return pathlib.Path(path)
    except:
        print('Could not open directory: ' + str(path))
        raise


def get_file_name(path):
    try:
        return str(pathlib.Path(path)).split('.')[0].split('/')[-1]
    except:
        print('Cannot parse a file path: ' + str(path))
        raise


def dicom_to_png_pkl(input, output, pickle):
    ds = load_dicom(os.path.normpath(input))
    # try to fetch the relevant data (so maybe we can process it later)
    dcm_data = []
    for field in fields_of_interest:
        try:
            dcm_data.append(getattr(ds, field))
        except:
            dcm_data.append('NULL')

    # a colormap and a normalization instance
    cmap = plt.cm.gray
    norm = plt.Normalize(vmin=ds.pixel_array.min(), vmax=ds.pixel_array.max())

    # map the normalized data to colors
    image = cmap(norm(ds.pixel_array))

    # check if the image/pickle output directory exists
    # if not create them
    create_dir_if_does_not_exist(output)
    create_dir_if_does_not_exist(pickle)

    # save the image
    try:
        plt.imsave(os.path.normpath(output), image, cmap='gray')
    except:
        print('Invalid image path')
        raise

    # create data frame to keep records of the images
    df = pd.DataFrame(dcm_data, fields_of_interest)

    # save dataframe for later usage
    try:
        df.to_pickle(os.path.normpath(pickle))
    except:
        print('Invalid pickle location')
        raise


# Process dicom files in the input_folder and stores result in the output_fodler
def process_dicom_from_to_folder(input_folder, output_folder):
    # Open specified input directory
    in_dir = open_directory(input_folder);
    # Process files in the input directory
    for path in in_dir.iterdir():
        if path.is_file():
            file_name = get_file_name(path)
            image_output = output_folder + '/images/' + file_name + '.png'
            pickle_output = output_folder + '/pickle/' + file_name + '.pkl'
            dicom_to_png_pkl(path, image_output, pickle_output)


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


def print_dicom_header(dicom_file_path):
    ds = load_dicom(dicom_file_path)
    for a in dir(ds):
        print('{} --- {}'.format(a, getattr(ds, a)))


# dir = "./dicom_objects/test/"
# l = []
# for file in os.listdir(dir):
#     if file.endswith(".dcm.gz"):
#         ds = load_dicom(dir+file)
#         l.append(getattr(ds, 'InstanceNumber'))
#
# print(sorted(l))
# print(len(l))

file_name = 'dicom_objects/test/325261597578315993471860132776680.dcm.gz'
# # dicom_get_tensor_and_label(file_name)
# load_dicom(file_name)
# plot_dicom(file_name)
# plot_dicom(file_name, cmap='viridis')
# plot_dicom(file_name, cmap='inferno')
# print_dicom_header(file_name)


# Body Part Examined
# Series Description
# Accesion Number is voor reverse lookup
# zit in series description