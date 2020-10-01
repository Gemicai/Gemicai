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
    return DataSet.get_dicomo_data_set(path, ['Modality'])


def demo_initialize_classifier():
    # Use resnet 18 as the base model for our new classifier
    resnet18 = models.resnet18(pretrained=True)
    net = gem.Classifier(resnet18, verbosity_level=2, enable_cuda=True)

    # Determine classes so the model knows number of it's outputs
    net.determine_classes(demo_get_dataset(train_data_set_path))
    net.save(classifier_path)


def demo_train_classifier():
    # Load a classifier from a file
    net = gem.Classifier.from_pickle(classifier_path)

    # Train the classifier
    # net.set_trainable_layers([("all", True)]) # by default all layers are trainable
    # net.set_device(enable_cuda=False)
    net.train(demo_get_dataset(train_data_set_path), num_workers=0, epochs=1, pin_memory=True)
    net.save(trained_classifier_path)


def demo_evaluate_classifier():
    net = gem.Classifier.from_pickle(trained_classifier_path)
    net.evaluate(demo_get_dataset(eval_data_set_path), num_workers=0, pin_memory=True)


def demo_create_dicomo_dataset():
    data_origin = '/mnt/data2/pukkaj/teach/study/PJ_TEACH/PJ_RESEARCH/'
    data_destination = '/home/nheinen/gemicai/dicom_objects/DX/'
    gem.create_dicomobject_dataset_from_folder(data_origin, data_destination, ['Modality'],
                                               field_values=[('Modality', ['DX'])])


def demo_things_we_should_fix():
    # FIXME: Exception handling in the data_iterators (Not extremely important rn but something we should consider)
    #  This raises a StopIteration exception, should be a FileNotFound or something like that.
    dataset = DataSet.get_dicomo_data_set('wrong_path')
    data_loader = torch.utils.data.DataLoader(dataset, 4)
    for data in data_loader:
        pass


# this has to wrap the code we call
# you can say thank you to how python implements multithreading
# and yes it has to be here and not in the Classifier.py
if __name__ == '__main__':
#    demo_prepare_data_set()
    demo_initialize_classifier()
    demo_train_classifier()
    demo_evaluate_classifier()
# demo_create_dicomo_dataset()
