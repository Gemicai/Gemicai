import os
import sys
# Makes sure we can import dicomo, and run this file from the terminal
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import classifier

data_origin = '/home/nheinen/gemicai/dicom_objects/DX/000001.dicomos.gz'
# data_origin = '../examples/compressed/DX/000001.dicomos.gz'


class Counter:
    def __init__(self):
        self.dic = {}

    def update(self, s):
        if s in self.dic.keys():
            self.dic[s] += 1
        else:
            self.dic[s] = 1

    def print(self):
        for k, v in self.dic.items():
            print('{} \t {}'.format(k, v))


cnt = Counter()
dataloader = classifier.get_data_loader(data_directory=data_origin)

for i, data in enumerate(dataloader):
    labels = data[0]
    for l in labels:
        cnt.update(l)
cnt.print()
print(cnt.dic.keys())



