import os
import pickle
import shutil

import torch
import gemicai as gem
import gzip
import tempfile
from itertools import count


def find_data():
    modalities = ['CT', 'MR', 'DX', 'MG', 'US', 'PT']
    modality = modalities[1]
    data_origin = '/mnt/data2/pukkaj/teach/study/PJ_TEACH/PJ_RESEARCH/'
    data_destination = '/home/nheinen/tsclient/niekh/Desktop/zgt/utilities/examples/dicom/' + modality + '/'

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


def construct_demo_data():
    # because of windows we have to manage temp file ourselves
    temp = tempfile.NamedTemporaryFile(mode="ab+", delete=False)

    try:
        # counts distinct field values
        cnt = gem.LabelCounter()
        # holds names for the gziped files
        filename_iterator = ("%06i.gemset" % i for i in count(1))
        objects_inside = 0
        with open('demo_data.pkl', 'rb') as inp:
            dataset = pickle.load(inp)
            for data in dataset.values():
                for d in data:
                    d = gem.DicomObject(tensor=d['tensor'], labels=['modality', 'bpe', 'studydes', 'seriesdes'],
                                        label_values=[d['modality'], d['bpe'], d['studydes'], d['seriesdes']])

                    pickle.dump(d, temp)


        temp.flush()
        zip_to_file(temp, next(filename_iterator))
    finally:
        temp.close()
        os.remove(temp.name)
    return cnt


def zip_to_file(file, zip_path):
    None
    with gzip.open(zip_path, 'wb') as zipped:
        shutil.copyfileobj(open(file.name, 'rb'), zipped)


construct_demo_data()
