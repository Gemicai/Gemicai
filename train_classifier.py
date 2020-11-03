from gemicai.data_iterators import DicomoDataset as DataSet
import torchvision.models as models
import gemicai as gem
import torch
import os
from datetime import datetime

dicom_fields = ['Modality', 'ImageType', 'ProtocolName', 'StudyDescription', 'SeriesDescription', 'BodyPartExamined']
path_input = os.path.join("examples", "dicom", "CT")
path_output = os.path.join("examples", "gzip", "CT")

train_dataset = os.path.join("examples", "gzip", "DX")
eval_dataset = os.path.join("examples", "gzip", "DX")
classifier_path = os.path.join("classifiers", "dx_bpe.gemclas")
trained_classifier_path = os.path.join("classifiers", "dx_bpe_trained.gemclas")


# train_dataset = '/mnt/SharedStor/datasets/dx/train/'
# test_dataset = '/mnt/SharedStor/datasets/dx/test/'
# classifier_path = '/mnt/SharedStor/classifiers/dx_bpe_trained.pkl'
# tree_path = '/mnt/SharedStor/classifiers/big_tree.pkl'


def demo_prepare_data_set():
    gem.dicom_to_gemset(path_input, path_output, dicom_fields, objects_per_file=25,
                        field_values=[('Modality', ['CT'])], pick_middle=False)


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
    testset = gem.DicomoDataset.get_dicomo_dataset(eval_dataset, labels=['BodyPartExamined'])
    # net.train(dataset, epochs=10, test_dataset=testset, verbosity=2)
    net.train(dataset, epochs=2, test_dataset=testset, verbosity=2,
              output_policy=gem.ToConsoleAndExcelFile("test.xlsx"))
    net.save(classifier_path)


def demo_evaluate_classifier():
    net = gem.Classifier.from_file(classifier_path)
    dataset = gem.DicomoDataset.get_dicomo_dataset(eval_dataset, labels=['BodyPartExamined'])
    # net.evaluate(dataset, verbosity=2)
    net.evaluate(dataset, verbosity=2, output_policy=gem.ToConsoleAndExcelFile("test.xlsx"))


def demo_create_dicomo_dataset():
    data_origin = '/mnt/data2/pukkaj/teach/study/PJ_TEACH/PJ_RESEARCH/'
    data_destination = '/home/nheinen/gemicai/dicom_objects/DX/'
    gem.dicom_to_gemset(data_origin, data_destination, ['Modality'],
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


def demo_classify_tensor():
    net = gem.Classifier.from_file(classifier_path)
    dataset = gem.DicomoDataset.get_dicomo_dataset(eval_dataset, labels=['BodyPartExamined'])
    data = next(iter(dataset))
    print(net.classify(data[0]))


input = os.path.join("examples", "dicom", "DX")
output = os.path.join("iterator_test")
output_new = os.path.join("iterator_test_2")

output_file = os.path.join("iterator_test", "000001.gemset")

output_file_test = os.path.join("iterator_test_2", "test", "000001.gemset")
output_file_train = os.path.join("iterator_test_2", "train", "000001.gemset")

def test():
    gem.dicom_to_gemset(input, output, dicom_fields, objects_per_file=25,
                        pick_middle=False)
    dataset = iter(gem.DicomoDataset.get_dicomo_dataset(output_file, labels=['Modality'],
                                                        constraints={'BodyPartExamined': ['VERTEBRAL COLUMN']}))

    dataset.save(output_new)
    dataset.summarize('BodyPartExamined')

    #dataset2 = gem.DicomoDataset.get_dicomo_dataset(os.path.join(output_new, "000001.gemset"), labels=['Modality'])
    #dataset2.summarize('BodyPartExamined')
    #dataset2.split(sets={os.path.join(output_new, 'train'): 0.2, os.path.join(output_new, 'test'): 0.8}, max_objects_per_file=1,
    #               self_erase_afterwards=True)

    #test_set = gem.DicomoDataset.get_dicomo_dataset(output_file_test, labels=['Modality'])
    #for obj in test_set:
    #    print(obj[1])
    #print("---------")

    #train_set = gem.DicomoDataset.get_dicomo_dataset(output_file_train, labels=['Modality'])
    #for obj in train_set:
    #    print(obj[1])


# this has to wrap the code we call
# you can say thank you to how python implements multithreading
# and yes it has to be here and not in the Classifier.py
if __name__ == '__main__':
    test()
    #demo_prepare_data_set()
    #demo_initialize_classifier()
    # demo_train_classifier()
    # demo_evaluate_classifier()
    #demo_classify_tensor()
    # demo_create_dicomo_dataset()
    # demo_initialize_tree()
    # demo_train_tree()
    # ds = demo_get_dataset()
