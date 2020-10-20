# This is the General-purpose medical image classification AI.
import pydicom
import gemicai as gem
import os


class Gemicai:
    def __init__(self, classifiers_path):
        self.relevant_modalities = ['CT', 'DX', 'MG', 'MR', 'PT', 'US']
        # self.relevant_modalities = ['DX', 'MG']
        self.classifiers_path = classifiers_path
        self.trees = {
            'DX': gem.ClassifierTree.from_dir(os.path.join(self.classifiers_path, 'dx_tree'))
        }

    def classify(self, dcm):
        assert isinstance(dcm, pydicom.Dataset), 'classify parameter should be of type pydicom.Dataset'
        modality = getattr(dcm, 'Modality')
        if modality not in self.relevant_modalities:
            raise Exception('Modality "{}" is not supported.'.format(modality))
        tensor = gem.extract_tensor(dcm)
        if modality == 'MG':
            return self.classify_mg(tensor)
        # For demonstration purposes use the DX tree anyway
        return self.trees['DX'].classify(tensor)

    def classify_mg(self, tensor):
        net = gem.Classifier.from_file(os.path.join(self.classifiers_path, 'mg', 'resnext.gemclas'))
        return {
            'Modality': [('MG', 1.0)],
            'BodyPartExamined': [('BREAST', 1.0)],
            'SeriesDescription': net.classify(tensor)[0]
        }
