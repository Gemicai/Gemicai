# This is the General-purpose medical image classification AI.
import pydicom
import gemicai as gem
import os
from abc import ABC, abstractmethod


# Use this class to build your own General-purpose medical image classification AI, or for short, Gemicai.
class Gemicai(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def classify(self, dcm):
        pass


# The ZGT implementation of a Gemicai
class GemicaiZGT(Gemicai):
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

        # All modalities just classify with DX tree for now, as im still training the trees. This way Kevin can use the
        # AI with deployment and we don't have to worry about new classifiers that should be added to the deploy etc.
        if modality == 'CT':
            return self._classify_mg(tensor)
        elif modality == 'DX':
            return self._classify_dx(tensor)
        elif modality == 'MG':
            return self._classify_mg(tensor)
        elif modality == 'MR':
            return self._classify_mr(tensor)
        elif modality == 'PT':
            return self._classify_pt(tensor)
        elif modality == 'US':
            return self._classify_us(tensor)

    def _classify_ct(self, tensor):
        return self.trees['DX'].classify(tensor)

    def _classify_mg(self, tensor):
        net = gem.Classifier.from_file(os.path.join(self.classifiers_path, 'mg', 'resnext.gemclas'))
        return {
            'SeriesDescription': net.classify(tensor)[0]
        }

    def _classify_dx(self, tensor):
        return self.trees['DX'].classify(tensor)

    def _classify_mr(self, tensor):
        return self.trees['DX'].classify(tensor)

    def _classify_pt(self, tensor):
        return self.trees['DX'].classify(tensor)

    def _classify_us(self, tensor):
        return self.trees['DX'].classify(tensor)
