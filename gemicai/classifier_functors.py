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
            raise TypeError("module parameter should have a base class of nn.Module")
        if not isinstance(classes, list):
            raise TypeError("classes parameter should be a list")
        # All popular models from https://pytorch.org/docs/stable/torchvision/models have either .fc or .classifier[-1].
        # as their final layer.
        try:
            module.fc = nn.Linear(module.fc.in_features, len(classes))
        except nn.modules.module.ModuleAttributeError:
            module.classifier[-1] = nn.Linear(module.classifier[-1].in_features, len(classes))
