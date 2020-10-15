import gemicai as gem
from datetime import datetime
import os


data_origin = '/mnt/data2/pukkaj/teach/study/PJ_TEACH/PJ_RESEARCH'
data_origin_just_DX = '/mnt/data2/pukkaj/teach/study/LBLPROJECT1/ZGT_LBLPROJECT1'

data_destination = '/mnt/SharedStor/eval_dataset'


dicom_fields = ['Modality', 'BodyPartExamined', 'StudyDescription', 'SeriesDescription']
relevant_modalities = ["CT", "MR", "US", "MG", "PT", "DX"]


def create_dataset(folder):
    start = datetime.now()
    gem.create_dicomobject_dataset_from_folder(data_origin, os.path.join(data_destination, folder), dicom_fields,
                                               field_values=[("Modality", [folder])], objects_per_file=100, pick_middle=True)
    print('create_dataset_with_common_modalities: Total time elapsed: {}'.format(str(datetime.now() - start)))


create_dataset("CT")
create_dataset("MR")
create_dataset("US")
create_dataset("MG")
create_dataset("PT")
create_dataset("DX")

