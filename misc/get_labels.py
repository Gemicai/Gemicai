import os
import sys
# Makes sure we can import dicomo, and run this file from the terminal
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import classifier

data_origin = '/home/nheinen/gemicai/dicom_objects/DX/000001.dicomos.gz'
# data_origin = '../examples/compressed/DX/000001.dicomos.gz'




cnt = Counter()
dataloader = classifier.get_data_loader(data_directory=data_origin)

for i, data in enumerate(dataloader):
    labels = data[0]
    for l in labels:
        cnt.update(l)
cnt.print()
print(cnt.dic.keys())



