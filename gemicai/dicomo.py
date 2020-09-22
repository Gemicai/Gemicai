from itertools import count
import pickle
import torchvision
import torch
import numpy
import os
from matplotlib import pyplot as plt
import tempfile
import gzip
import shutil

from gemicai import dicom_utilities as du


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


# All files within the origin directory will be compressed, returns counter for the frequency of bpe label.
# fixme: this doesn't work on windows bc of tempfile.NamedTemporaryFile.
# (at this point in time not really worth fixing, bc who cares about windows anyway?)
def compress_dicom_files(origin, destination, objects_per_file=1000):
    # Relevant modalities
    modalities = ['CT', 'MR', 'DX', 'MG', 'US', 'PT']
    # Trying just the DX modality first, as that's probably the easist one.
    modalities = ['DX']
    cnt = LabelCounter()
    with tempfile.NamedTemporaryFile(mode="ab+") as temp:
        # holds names for the gziped files
        filename_iterator = ("%06i.dicomos.gz" % i for i in count(1))
        objects_inside = 0

        for root, dirs, files in os.walk(origin):
            for file in files:
                try:
                    d = Dicomo(root + '/' + file)
                    if d.modality in modalities:
                        cnt.update(d.bpe)
                        # check if we are not allowed to append more files
                        if objects_inside >= objects_per_file:
                            # gzip temp file and clear its content
                            temp.flush()
                            zip_to_file(temp, destination + next(filename_iterator))
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
        zip_to_file(temp, destination + next(filename_iterator))
        return cnt


def zip_to_file(file, zip_path):
    with gzip.open(zip_path, 'wb') as zipped:
        shutil.copyfileobj(open(file.name, 'rb'), zipped)


def unzip_to_file(file, zip_path):
    with gzip.open(zip_path, 'rb') as zipped:
        shutil.copyfileobj(zipped, open(file.name, 'ab+'))


def stream_pickles(path):
    with tempfile.NamedTemporaryFile(mode="ab+") as file:
        try:
            unzip_to_file(file, path)
            while True:
                yield pickle.load(file)
        except EOFError:
            pass


# Putting this here since standard collection.Counter doesn't do what I want it to do.
class LabelCounter:
    def __init__(self):
        self.dic = {}

    def update(self, s):
        if s in self.dic.keys():
            self.dic[s] += 1
        else:
            self.dic[s] = 1

    # I know this looks hideous but it prints a wonderfull table :)
    def print(self):
        print('label                | frequency\n---------------------------------')
        t = 0
        for k, v in self.dic.items():
            t += v
            print('{:<20s} | {:>8d}'.format(k, v))
        print('\nTotal number of training images: {} \nTotal number of labels: {}'.format(t, len(self.dic.keys())))
