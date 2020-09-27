from abc import ABC, abstractmethod
import torch.nn as nn


class GEMICAIABCFunctor(ABC):
    def __init__(self):
        None

    # should configure the last layer of the model
    @abstractmethod
    def __call__(self, model):
        pass


class DefaultLastLayerConfig(GEMICAIABCFunctor):

    def __call__(self, module, classes):
        if not isinstance(module, nn.Module):
            raise Exception("module parameter should have a base class of nn.Module")
        if not isinstance(classes, list):
            raise Exception("classes parameter should be a list")
        module.fc = nn.Linear(module.fc.in_features, len(classes))
        # You don't have to return the object, just calling it is sufficient
        # return module
