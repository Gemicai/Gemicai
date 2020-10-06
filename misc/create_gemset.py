from .. import gemicai as gem
from datetime import datetime

start = datetime.now()

data_origin = '/mnt/data2/pukkaj/teach/study/PJ_TEACH/PJ_RESEARCH/'
data_origin = '/mnt/data2/pukkaj/teach/study/LBLPROJECT1/ZGT_LBLPROJECT1/'
data_destination = '/mnt/SharedStor/datasets/dx/train/'
dicom_fields = ['Modality', 'BodyPartExamined', 'StudyDescription', 'SeriesDescription']

gem.create_dicomobject_dataset_from_folder(data_origin, data_destination, dicom_fields,
                                           field_values=[('Modality', ['DX'])])

print('Total time elapsed: {}'.format(str(datetime.now() - start)))
