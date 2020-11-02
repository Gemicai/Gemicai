import gemicai as gem
from datetime import datetime
import os
import shutil


data_origin = '/mnt/data2/pukkaj/teach/study/PJ_TEACH/PJ_RESEARCH'
data_origin_just_DX = '/mnt/data2/pukkaj/teach/study/LBLPROJECT1/ZGT_LBLPROJECT1'

data_destination = '/mnt/SharedStor/tutorials/Mammography'


dicom_fields = ['Modality', 'BodyPartExamined', 'StudyDescription', 'SeriesDescription']
relevant_modalities = ["CT", "MR", "US", "MG", "PT", "DX"]


def create_dataset(folder):
    start = datetime.now()
    gem.dicom_to_gemset(data_origin, os.path.join(data_destination, folder), dicom_fields,
                        field_values=[("Modality", [folder])], objects_per_file=100, pick_middle=True)
    print('create_dataset_with_common_modalities: Total time elapsed: {}'.format(str(datetime.now() - start)))


# Both origin and destination are directories
def copy_files(origin, destination):
    for root, dirs, files in os.walk(origin):
        for file in files:
            try:
                d = gem.DicomObject.from_file(root + '/' + file, ['Modality'], tensor_size=(244, 244))
                if d.get('Modality') == 'MG':
                    shutil.copyfile(root + '/' + file, destination + '/' + file)
            except:
                pass


copy_files(data_origin, data_destination)
# create_dataset("CT")
# create_dataset("MR")
# create_dataset("US")
# create_dataset("MG")
# create_dataset("PT")
# create_dataset("DX")

