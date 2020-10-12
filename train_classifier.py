from gemicai.data_iterators import DicomoDataset as DataSet
import torchvision.models as models
import gemicai as gem
import torch
import os
from datetime import datetime

dicom_fields = ['Modality', 'ImageType', 'ProtocolName', 'StudyDescription', 'SeriesDescription', 'BodyPartExamined']
path_input = os.path.join("examples", "dicom", "CT")
path_output = os.path.join("examples", "gzip", "CT")

train_data_set_path = os.path.join("examples", "gzip", "CT")
eval_data_set_path = os.path.join("examples", "gzip", "CT")
classifier_path = os.path.join("classifiers", "dx_bpe.pkl")
trained_classifier_path = os.path.join("classifiers", "dx_bpe_trained.pkl")

train_dataset = '/mnt/SharedStor/datasets/dx/train/'
test_dataset = '/mnt/SharedStor/datasets/dx/test/'
classifier_path = '/mnt/SharedStor/classifiers/dx_bpe_trained.pkl'
tree_path = '/mnt/SharedStor/classifiers/big_tree.pkl'


def demo_prepare_data_set():
    gem.create_dicomobject_dataset_from_folder(path_input, path_output, dicom_fields, objects_per_file=25,
                                               field_values=[('Modality', ['CT'])])


def demo_get_dataset(path):
    return DataSet.get_dicomo_dataset(path, ['Modality'])


def demo_initialize_classifier():
    # Use resnet 18 as the base model for our new classifier
    resnet18 = models.resnet18(pretrained=True)
    dataset = gem.DicomoDataset.get_dicomo_dataset(train_dataset, labels=['BodyPartExamined'])
    dataset.summarize('BodyPartExamined')
    net = gem.Classifier(resnet18, dataset.classes('BodyPartExamined'), enable_cuda=True)
    net.save(classifier_path)


def demo_train_classifier():
    # Load a classifier from a file
    net = gem.Classifier.from_file(classifier_path)
    dataset = gem.DicomoDataset.get_dicomo_dataset(train_dataset, labels=['BodyPartExamined'])
    # Train the classifier
    # net.set_trainable_layers([("all", True)]) # by default all layers are trainable
    # net.set_device(enable_cuda=False)
    # net.train(dataset, num_workers=0, epochs=1, pin_memory=True)

    # Train with evaluation dataset
    testset = gem.DicomoDataset.get_dicomo_dataset(test_dataset, labels=['BodyPartExamined'])
    net.train(dataset, epochs=10, test_dataset=testset, verbosity=2)
    # net.train(dataset, epochs=10, test_dataset=testset, verbosity=2, output_policy=gem.OutputToExcelFile("test.xlsx"))
    net.save(classifier_path)


def demo_evaluate_classifier():
    net = gem.Classifier.from_file(classifier_path)
    dataset = gem.DicomoDataset.get_dicomo_dataset(test_dataset, labels=['BodyPartExamined'])
    net.evaluate(dataset, verbosity=2)
    # net.evaluate(dataset, verbosity=2, output_policy=gem.OutputToExcelFile("test.xlsx"))


def demo_create_dicomo_dataset():
    data_origin = '/mnt/data2/pukkaj/teach/study/PJ_TEACH/PJ_RESEARCH/'
    data_destination = '/home/nheinen/gemicai/dicom_objects/DX/'
    gem.create_dicomobject_dataset_from_folder(data_origin, data_destination, ['Modality'],
                                               field_values=[('Modality', ['DX'])])


def demo_initialize_tree():
    # Select the fields the tree should sequencially. The first item in the list will what the root of the tree
    # classifies. Initializing the Tree takes quite long, as it has to calculate the classes for every node.
    relevant_labels = ['Modality', 'BodyPartExamined', 'StudyDescription', 'SeriesDescription']
    resnet18 = models.resnet18(pretrained=True)
    ds = gem.DicomoDataset.get_dicomo_dataset(train_dataset, labels=relevant_labels)
    net = gem.Classifier(resnet18, [], enable_cuda=True)

    start = datetime.now()
    tree = gem.ClassifierTree(default_classifier=net, labels=relevant_labels, base_dataset=ds)
    print('Initializing tree took : {}'.format(gem.utils.strfdelta(datetime.now() - start, '%H:%M:%S')))
    tree.save(tree_path)


def demo_train_tree():
    tree = gem.ClassifierTree.from_file(tree_path)
    print(tree)


# this has to wrap the code we call
# you can say thank you to how python implements multithreading
# and yes it has to be here and not in the Classifier.py
if __name__ == '__main__':
    # demo_prepare_data_set()
    # demo_initialize_classifier()
    # demo_train_classifier()
    # demo_evaluate_classifier()
    # demo_create_dicomo_dataset()
    demo_initialize_tree()
    demo_train_tree()
    # ds = demo_get_dataset()


