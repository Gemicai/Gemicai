import os
import sys
from collections import Counter

# Makes sure we can import dicomo, and run this file from the terminal
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import PickleDataSet
import classifier

data_origin = '/home/nheinen/gemicai/dicom_objects/DX/000001.dicomos.gz'
# data_origin = '../examples/compressed/DX/000001.dicomos.gz'


pds = PickleDataSet.PickleDataSet(data_origin, ['tensor', 'bpe'])
pds_iter = iter(pds)
cnt = Counter()
for _ in range(1000):
    label = next(pds_iter)[1]
    cnt.update(label)

for value, count in cnt.most_common():
    print(value, count)


# This should work but doesn't
# dataloader = classifier.get_data_loader(data_directory=data_origin)
#
# for i, data in enumerate(dataloader):
#     # get the inputs; data is a list of [inputs, labels]
#     tensor, labels = data
#     print(data)


