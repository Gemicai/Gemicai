"""This module contains functors that can be used to modify models. Object of this class can be passed as an optional
argument to the gemicai.Classifier.Classifier constructor, see it's layer_config parameter for more information."""

from abc import ABC, abstractmethod
import torch.nn as nn


class GEMICAIABCFunctor(ABC):
    """Every custom functor should extend this abstract base class."""

    def __init__(self):
        None

    @abstractmethod
    def __call__(self, model):
        """A call to this function should eg configure the last layer of the model."""
        pass


class DefaultLastLayerConfig(GEMICAIABCFunctor):
    """Gemicai's default functor which modifies model's final layer depending on the number of passed classes.
    It works with most of the torchvision models, for more information about models themselves please refer
    to the https://pytorch.org/docs/stable/torchvision/models"""

    def __call__(self, model, classes):
        """Modifies model's final layer depending on the number of passed classes.

        :param model: model to modify
        :type model: nn.Module
        :param classes: a list of classes present in the used dataset
        :type classes: list
        """
        if not isinstance(model, nn.Module):
            raise TypeError("module parameter should have a base class of nn.Module")
        if not isinstance(classes, list):
            raise TypeError("classes parameter should be a list")
        # All popular models from https://pytorch.org/docs/stable/torchvision/models have either .fc or .classifier[-1].
        # as their final layer.
        try:
            model.fc = nn.Linear(model.fc.in_features, len(classes))
        except nn.modules.module.ModuleAttributeError:
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(classes))
