import os
import sys

# Makes sure we can import classifier, and run this file from the terminal
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import classifier
import torchvision.models as models

# Use resnet 18 as the base model for our new classifier
resnet18 = models.resnet18(pretrained=True)
classifier = classifier.Classifier(resnet18)

classifier.set_data_loader('/home/nheinen/gemicai/dicom_objects/DX/')
classifier.train(epochs=20, verbosity=1)
classifier.evaluate('/home/nheinen/gemicai/dicom_objects/DX/')