import os
import sys

# Makes sure we can import dicomo, and run this file from the terminal
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dicomo

data_origin = '/mnt/data2/pukkaj/teach/study/PJ_TEACH/PJ_RESEARCH/'
data_destination = '/home/nheinen/gemicai/dicom_objects/'
dicomo.compress_dicom_files(data_origin, data_destination)

# l = dicomo.load_object('../examples/compressed/CT/000001')
# dicomo.plot_dicomo(l[42])
