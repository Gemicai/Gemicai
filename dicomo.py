from itertools import count
#from compress_pickle import dump, load
import pickle
import torchvision
import torch
import numpy
import os
import dicom_utilities as du
from matplotlib import pyplot as plt



# Dicom object, used to extract only the relevant data (for training) from a dicom file.
class Dicomo:
    def __init__(self, filename):
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

        self.tensor = create_tensor(data)[0]
        self.bpe = get_attr(ds, 'BodyPartExamined')
        self.seriesdes = get_attr(ds, 'SeriesDescription')
        self.studydes = get_attr(ds, 'StudyDescription')
        self.modality = get_attr(ds, 'Modality')
        self.imtype = get_attr(ds, 'ImageType')
        self.protocol = get_attr(ds, 'ProtocolName')


# Because getattr() trhows an AttributeError if the field is left empty in the dicom header
def get_attr(ds, attr):
    try:
        return getattr(ds, attr)
    except AttributeError:
        return None


# Plots dicom image with some additional label info.
def plot_dicomo(d: Dicomo, cmap='gray'):
    plt.title('{} | {} | {} | {} \n {} | {}'.format(d.modality, d.bpe, d.studydes, d.seriesdes, d.imtype, d.protocol))
    plt.imshow(d.tensor, cmap)
    plt.show()


# All files within the origin directory will be compressed.
def compress_dicom_files(origin, destination, objects_per_file=1000):
    filename_iterator = ("%06i" % i for i in count(1))
    filename = next(filename_iterator)
    objects_inside = 0

    # Relevant modalities
    modalities = ['CT', 'MR', 'DX', 'MG', 'US', 'PT']
    for root, dirs, files in os.walk(origin):
        for file in files:
            try:

                d = Dicomo(root + '/' + file)
                if d.modality in modalities:
                    # check if we are allowed to append more files
                    # if not get next file
                    if objects_inside >= objects_per_file:
                        filename = next(filename_iterator)
                        objects_inside = 0

                    objects_inside += 1
                    dump_to_pickle(d, destination+filename)

            except Exception as ex:
                template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                message = template.format(type(ex).__name__, ex.args)
                print(message)

# TODO DISCUSS COMPRESSION
def dump_to_pickle(obj, output_file):
    with open(output_file, 'ab') as output:
        pickle.dump(obj, output)
        #dump(obj, output, compression="lzma", set_default_extension=False)

def stream_pickles(path):
    with open(path, 'rb') as file:
        try:
            while True:
                #yield load(file, compression="lzma", set_default_extension=False)
                yield pickle.load(file)
        except EOFError:
            pass

