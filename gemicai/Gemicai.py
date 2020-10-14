# This is the General-purpose medical image classification AI.
import pydicom
import gemicai as gem


class Gemicai:
    def __init__(self):
        # self.relevant_modalities = ['CT', 'DX', 'MG', 'MR', 'PT', 'US']
        self.relevant_modalities = ['DX']
        self.trees = {
            'DX': gem.ClassifierTree.from_dir('tests/bench/examples/tree/')
        }

    def classify(self, dcm):
        assert isinstance(dcm, pydicom.Dataset), 'classify parameter should be of type pydicom.Dataset'
        modality = getattr(dcm, 'Modality')
        if modality not in self.relevant_modalities:
            raise Exception('Modality "{}" is not supported.'.format(modality))
        tensor = gem.extract_tensor(dcm)
        return self.trees[modality].classify(tensor)
