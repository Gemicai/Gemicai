import os
import sys

# Makes sure we can import dicomo, and run this file from the terminal
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import PickleDataSet
import torchvision
import classifier

data_origin = '/home/nheinen/gemicai/dicom_objects/DX/'

# This should work but doesn't
# dataloader = classifier.get_data_loader(data_directory=data_origin)
#
# for i, data in enumerate(dataloader, 0):
#     # get the inputs; data is a list of [inputs, labels]
#     tensor, labels = data
#     print(data)

transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Grayscale(3),
        torchvision.transforms.ToTensor()
    ])
pds = PickleDataSet.PickleDataSet(data_origin, ['tensor', 'bpe'], transform)
for _ in range(100):
    print(next(pds))
