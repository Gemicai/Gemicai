# import dicom_utilities as du
import os
from itertools import count
filename = ("data_%08i.pkl" % i for i in count(1))

data_origin = 'C:/Users/niekh/Desktop/zgt'
os.chdir(data_origin)


# for root, dirs, files in os.walk('.'):
#     for file in files
