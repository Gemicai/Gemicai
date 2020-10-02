from gemicai.data_iterators import DicomoDataset as DataSet
import torchvision.models as models
import gemicai as gem
import torch
import os

dicom_fields = ['Modality', 'ImageType', 'ProtocolName', 'StudyDescription', 'SeriesDescription', 'BodyPartExamined']
path_input = os.path.join("examples", "dicom", "CT")
path_output = os.path.join("examples", "gzip", "CT")

train_data_set_path = os.path.join("examples", "gzip", "CT")
eval_data_set_path = os.path.join("examples", "gzip", "CT")
classifier_path = os.path.join("classifiers", "dx_bpe.pkl")
trained_classifier_path = os.path.join("classifiers", "dx_bpe_trained.pkl")

train_dataset = '/mnt/SharedStor/datasets/dx/train/'
test_dataset = '/mnt/SharedStor/datasets/dx/test/'
# classifier_path = '/mnt/SharedStor/classifiers/dx_bpe.pkl'


def demo_prepare_data_set():
    gem.create_dicomobject_dataset_from_folder(path_input, path_output, dicom_fields, objects_per_file=25,
                                               field_values=[('Modality', ['CT'])])


def demo_get_dataset(path):
    return DataSet.get_dicomo_dataset(path, ['Modality'])


def demo_initialize_classifier():
    # Use resnet 18 as the base model for our new classifier
    resnet18 = models.resnet18(pretrained=True)
    net = gem.Classifier(resnet18, verbosity_level=2, enable_cuda=True)

    # Determine classes so the model knows number of it's outputs
    net.determine_classes(demo_get_dataset(train_data_set_path))
    net.save(classifier_path)


def demo_train_classifier():
    # Load a classifier from a file
    net = gem.Classifier.load(classifier_path)

    # Train the classifier
    # net.set_trainable_layers([("all", True)]) # by default all layers are trainable
    # net.set_device(enable_cuda=False)
    net.train(demo_get_dataset(train_data_set_path), num_workers=0, epochs=1, pin_memory=True)
    net.save(trained_classifier_path)


def demo_evaluate_classifier():
    net = gem.Classifier.load(trained_classifier_path)
    net.evaluate(demo_get_dataset(eval_data_set_path), num_workers=0, pin_memory=True)


def demo_create_dicomo_dataset():
    data_origin = '/mnt/data2/pukkaj/teach/study/PJ_TEACH/PJ_RESEARCH/'
    data_destination = '/home/nheinen/gemicai/dicom_objects/DX/'
    gem.create_dicomobject_dataset_from_folder(data_origin, data_destination, ['Modality'],
                                               field_values=[('Modality', ['DX'])])


# this has to wrap the code we call
# you can say thank you to how python implements multithreading
# and yes it has to be here and not in the Classifier.py
# if __name__ == '__main__':
# #    demo_prepare_data_set()
#     demo_initialize_classifier()
#     demo_train_classifier()
#     demo_evaluate_classifier()
# # demo_create_dicomo_dataset()

#ds = demo_get_dataset()

# For this example, we take resnet18
resnet18 = models.resnet18(pretrained=True)

dataset = gem.DicomoDataset.get_dicomo_dataset('misc/demo.gemset', ['bpe'])

# All a classifier needs is a base model, and a list of classes you want to classify
net = gem.Classifier(resnet18, dataset.classes('bpe'))

net.train(dataset, epochs=1, verbosity=1)
net.evaluate(dataset, verbosity=1)