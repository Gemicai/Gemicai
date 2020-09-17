import os
import sys

# Makes sure we can import dicomo, and run this file from the terminal
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dicomo

dataloader = dicomo.get_data_loader('/home/nheinen/gemicai/dicom_objects/DX/')

for i, data in enumerate(dataloader, 0):
    # get the inputs; data is a list of [inputs, labels]
    tensor, labels = data
    print(data)
