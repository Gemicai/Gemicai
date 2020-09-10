
import os
from itertools import count
import pickle
import dicom

filename = ("dicom_objects_%08i.pkl" % i for i in count(1))

data_origin = '/mnt/data2/pukkaj/teach/study/PJ_TEACH/PJ_RESEARCH/'
data_destination = '/home/nheinen/gemicai/dicom_objects/'
os.chdir(data_origin)

modalities = ['CT', 'MR', 'DX', 'MG', 'US', 'PT']
temp_list = []
attribute_errors = 0

for root, dirs, files in os.walk('.'):
    for file in files:
        try:
            d = dicom.Dicom(root+'/'+file)
            if d.modality in modalities:
                temp_list.append(d)
                if len(temp_list) >= 1000:
                    with open(data_destination+next(filename), 'wb') as output:
                        pickle.dump(temp_list, output, pickle.HIGHEST_PROTOCOL)
                        temp_list = []
        except AttributeError:
            attribute_errors += 1
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)

print('Finished ! Total attribute errors: {}'.format(attribute_errors))
