import gzip
import pydicom as dicom
import os
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
import sys
import string

image_formats = ["png", "jpg"]

# if needed we can export it later
fields_of_interest = ['Rows', 'StudyDate', 'SeriesTime', 'ContentTime', 'StudyInstanceUID', 'SeriesInstanceUID',
                      'SOPInstanceUID',
                      'Modality', 'SeriesDate', 'AccessionNumber', 'BodyPartExamined', 'StudyDescription',
                      'SeriesDescription', 'InstanceNumber',
                      'PatientOrientation', 'ImageLaterality', 'ImageComments', 'SeriesNumber', 'PatientName']


def load_dicom(filename):
    if (filename.endswith('.dcm')):
        ds = dicom.dcmread(filename)
    else:
        with gzip.open(filename) as fd:
            ds = dicom.dcmread(fd, force=True)
    ds.file_meta.TransferSyntaxUID = dicom.uid.ImplicitVRLittleEndian
    return ds


def get_directory_path(path):
    index = path.rfind('/')
    return os.path.normpath(path[0:index])


def create_dir_if_does_not_exist(path):
    dir_name = get_directory_path(path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


# ------------------------ END OF THE DECLARATIONS ------------------------

args = []
for arg in sys.argv:
    args.append(str(arg))

args = ['', 'examples/325261321823915159826883421451323.dcm.gz', 'examples/img1.png', 'examples/ok.pkl']
# process command line params
try:
    if len(args) != 4 or not (args[2].split('.')[1] in image_formats):
        raise Exception("Improper program usage");
except:
    print("Proper program usage: ")
    print("extract_data.py {input_location} {output_location}.{png/jpg} {pickle_location}.pkl")
    sys.exit(1)

# load a dcm file
try:
    ds = load_dicom(os.path.normpath(args[1]))
except:
    print("Please priovide a valid .dcm or .dcm.gz file")
    sys.exit(1)

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
create_dir_if_does_not_exist(args[2])
create_dir_if_does_not_exist(args[3])

# save the image
try:
    plt.imsave(os.path.normpath(args[2]), image, cmap="gray")
except:
    print("Invalid image path")
    sys.exit(1)

# create data frame to keep records of the images
df = pd.DataFrame(dcm_data, fields_of_interest)

# save dataframe for later usage
try:
    df.to_pickle(os.path.normpath(args[3]))
except:
    print("Invalid pickle location")
    sys.exit(1)
