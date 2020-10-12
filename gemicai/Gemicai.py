# This is the General-purpose medical image classification AI.
# For now it just returns the actual fields in the Dicom header, this way Kevin can already set up the server.

import pydicom
import gemicai as gem


class Gemicai:
    def __init__(self):
        pass
        # self.ct = gem.ClassifierTree()

    def classify(self, dcm):
        assert isinstance(dcm, pydicom.Dataset), 'classify parameter should be of type pydicom.Dataset'
        tensor = gem.extract_tensor(dcm)
        modality = getattr(dcm, 'Modality')
        bpe = getattr(dcm, 'BodyPartExamined')
        studydes = getattr(dcm, 'StudyDescription')
        seriesdes = getattr(dcm, 'SeriesDescription')
        orientation = 'Axial'
        return {
            'modality': [
                (modality, 0.914),
                ('EX', 0.08),
                ('EZ', 0.006),
            ],
            'bpe': [
                (bpe, 0.872),
                ('ABDOMEN', 0.12),
                ('HAND', 0.008),
            ],
            'studydes': [
                (studydes, 0.792),
                ('example 2', 0.189),
                ('example 3', 0.023),
            ],
            'seriesdes': [
                (seriesdes, 0.792),
                ('example 2', 0.189),
                ('example 3', 0.023),
            ],
            'orientation': [
                (orientation, 0.792),
                ('Frontal', 0.189),
                ('Lateral', 0.023),
            ],
        }
