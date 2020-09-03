import gzip
import pydicom as dicom
import os
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
import sys
import string
import pathlib

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
        print("Please priovide a valid .dcm or .dcm.gz file")
        sys.exit(1)
    return ds


def get_directory_path(path):
    index = path.rfind('/')
    return os.path.normpath(path[0:index])


def create_dir_if_does_not_exist(path):
    dir_name = get_directory_path(path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def open_directory(path, dir_type):
    try:
        return pathlib.Path(path)
    except:
        print("Could not open a specified " + dir_type + " directory")
        sys.exit(1)


def get_file_name(path):
    try:
        return str(path).split('.')[0].split('\\')[-1]
    except:
        print("Cannot parse a file path: " + str(path))
        sys.exit(1)


def extract_png(input, output, pickle):
    ds = load_dicom(os.path.normpath(input))
    # try to fetch the relevant data (so maybe we can process it later)
    dcm_data = []
    for field in fields_of_interest:
        try:
            dcm_data.append(getattr(ds, field))
        except:
            dcm_data.append("NULL")

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
        plt.imsave(os.path.normpath(output), image, cmap="gray")
    except:
        print("Invalid image path")
        sys.exit(1)

    # create data frame to keep records of the images
    df = pd.DataFrame(dcm_data, fields_of_interest)

    # save dataframe for later usage
    try:
        df.to_pickle(os.path.normpath(pickle))
    except:
        print("Invalid pickle location")
        sys.exit(1)


def extract_pngs(input_folder, output_folder):
    # Open specified input directory
    in_dir = open_directory(input_folder, "input");
    # Process files in the input directory
    for path in in_dir.iterdir():
        if path.is_file():
            file_name = get_file_name(path)
            image_output = output_folder + '/images/' + file_name + '.png'
            pickle_output = output_folder + '/pickle/' + file_name + '.pkl'
            extract_png(path, image_output, pickle_output)


# Returns ({imgage as tensor}, {label})
def extract_img_and_label(dicom_file_path):
    df = load_dicom(dicom_file_path)
    # For the neural network, the image needs to be represented as a tensor
    # TODO: implement this
    tensor = dicom_file_path
    label = getattr(df, 'BodyPartExamined')
    return tensor, label


extract_pngs('examples', 'examples/pngs')
print(extract_img_and_label('examples/1.dcm.gz'))
print(extract_img_and_label('examples/2.dcm.gz'))
print(extract_img_and_label('examples/3.dcm.gz'))
print(extract_img_and_label('examples/4.dcm.gz'))
print(extract_img_and_label('examples/5.dcm.gz'))