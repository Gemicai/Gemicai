import dicom_utilities as du
import os
from itertools import count
import pickle

filename = ("dicom_objects_%08i.pkl" % i for i in count(1))

data_origin = '/mnt/data2/pukkaj/teach/study/PJ_TEACH/PJ_RESEARCH/'
data_destination = '/home/nheinen/gemicai/dicom_objects/dx'
os.chdir(data_origin)

modalities = ['CT', 'MR', 'DX', 'MG', 'US', 'PT']
temp_list = []

for root, dirs, files in os.walk('.'):
    for file in files:
        try:
            dicom = du.Dicom(file)
            if dicom.modality in modalities:
                temp_list.append(dicom)
                if len(temp_list) >= 1000:
                    with open(next(filename), 'wb') as output:
                        pickle.dump(temp_list, output, pickle.HIGHEST_PROTOCOL)
                        temp_list = []
        except:
            print('Exception loading {} in folder {}'.format(file, root))


