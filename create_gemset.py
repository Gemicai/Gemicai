import gemicai as gem
from datetime import datetime


data_origin = '/mnt/data2/pukkaj/teach/study/PJ_TEACH/PJ_RESEARCH/'
data_origin_just_DX = '/mnt/data2/pukkaj/teach/study/LBLPROJECT1/ZGT_LBLPROJECT1/'

data_destination_all = '/mnt/SharedStor/datasets_new/all/'
data_destination_pick_middle = '/mnt/SharedStor/datasets_new/pick_middle/'

dicom_fields = ['Modality', 'BodyPartExamined', 'StudyDescription', 'SeriesDescription']


def create_dataset_with_pick_middle():
    start = datetime.now()
    gem.create_dicomobject_dataset_from_folder(data_origin, data_destination_pick_middle, dicom_fields,
                                               pick_middle=True)
    print('create_dataset_with_pick_middle: Total time elapsed: {}'.format(str(datetime.now() - start)))


def create_dataset():
    start = datetime.now()
    gem.create_dicomobject_dataset_from_folder(data_origin, data_destination_all, dicom_fields)
    print('create_dataset: Total time elapsed: {}'.format(str(datetime.now() - start)))


create_dataset_with_pick_middle()
create_dataset()

