import os
import torch
import gemicai as gem
from train_classifier import get_data_set


def find_data():
    modalities = ['CT', 'MR', 'DX', 'MG', 'US', 'PT']
    modality = modalities[1]
    data_origin = '/mnt/data2/pukkaj/teach/study/PJ_TEACH/PJ_RESEARCH/'
    data_destination = '/home/nheinen/tsclient/niekh/Desktop/zgt/utilities/examples/dicom/'+modality+'/'

    for root, dirs, files in os.walk(data_origin):
        for file in files:
            try:
                d = gem.Dicomo(root + '/' + file)
                if d.modality == modality:
                    gem.plot_dicomo(d)
            except Exception as ex:
                template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                message = template.format(type(ex).__name__, ex.args)
                print(message)
        print(root)
        if input('Want to continue looking? y/n') == 'n':
            break

    print(data_destination)


def construct_demo_dataset():
    data_origin = '../examples/gzip/dx/test/'
    dataset = get_data_set(data_origin)
    print(dataset)
    batch_size = 4
    data_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=False)
    demo_data = {
        'SKULL': [],
        'CHEST': [],
        'PELVIS': [],
        'HAND': [],
        'FOOT': []
    }
    for tensors, bpes in data_loader:
        for i in range(batch_size):
            if bpes[i] in demo_data.keys():
                if len(demo_data[bpes[i]]) < 20:
                    demo_data[bpes[i]].append(tensors[i])





