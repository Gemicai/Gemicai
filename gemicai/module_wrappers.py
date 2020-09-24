from abc import ABC, abstractmethod
import torch.nn as nn


# A hack since we cannot reliably steal nn.Module content from a child class
# and we still have to somehow update input features
class GEMICAIABCModuleWrapper(ABC):
    def __init__(self, module):
        if not isinstance(module, nn.Module):
            raise Exception("GEMICAIABCModule.__init__ expects module to be an instance of "
                            "nn.Module where as it is " + str(type(module)))
        self.module = module

    @abstractmethod
    def update_modules_input_features(self, classes):
        pass


class ModuleWrapper(GEMICAIABCModuleWrapper):

    def update_modules_input_features(self, classes):
        if not isinstance(classes, list):
            raise Exception("classes parameter should be a list")
        self.module.fc = nn.Linear(self.module.fc.in_features, len(classes))
