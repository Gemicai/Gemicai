import os
import sys
from datetime import datetime

# Makes sure we can import dicomo, and run this file from the terminal
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gemicai import dicomo

start = datetime.now()

data_origin = '/mnt/data2/pukkaj/teach/study/PJ_TEACH/PJ_RESEARCH/'
data_destination = '/home/nheinen/gemicai/dicom_objects/DX/'

# data_origin = 'C:/Users/niekh/Desktop/zgt/utilities/examples/dicom/CT/'
# data_destination = 'C:/Users/niekh/Desktop/zgt/utilities/examples/compressed/CT/'

cnt = dicomo.compress_dicom_files(data_origin, data_destination)
cnt.print()

print('Total time elapsed: {}'.format(str(datetime.now() - start)))




