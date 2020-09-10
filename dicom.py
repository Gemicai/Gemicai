
from matplotlib import pyplot as plt

import torchvision
import torch
import numpy

import dicom_utilities as du


class Dicom:
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
        self.bpe = getattr(ds, 'BodyPartExamined')
        self.seriesdes = getattr(ds, 'SeriesDescription')
        self.studydes = getattr(ds, 'StudyDescription')
        self.modality = getattr(ds, 'Modality')
        self.imtype = getattr(ds, 'ImageType')


# Plots dicom image with some additional label info.
def plot_dicom(d: Dicom, cmap='gray'):
    plt.title('{} | {} | {} | {} \n {}'.format(d.modality, d.bpe, d.studydes, d.seriesdes, d.imtype))
    plt.imshow(d.tensor, cmap)
    plt.show()


file = 'dicom_objects/test/325261597578315993471860132776680.dcm.gz'
d = Dicom(file)
plot_dicom(d)
