import torchvision.models as models
import gemicai as gem
import torch
import time
import os

train_data_set_path = os.path.join("examples", "gzip", "CT")
eval_data_set_path = os.path.join("examples", "gzip", "CT")
classifier_path = os.path.join("classifiers", "dx_bpe.pkl")
trained_classifier_path = os.path.join("classifiers", "dx_bpe_trained.pkl")

train_dataset = '/mnt/SharedStor/datasets/dx/train/'
test_dataset = '/mnt/SharedStor/datasets/dx/test/'
classifier_path = '/mnt/SharedStor/classifiers/dx_bpe.pkl'

def demo_initialize_classifier():
    # Use resnet 18 as the base model for our new classifier
    resnet18 = models.resnet18(pretrained=True)
    net = gem.Classifier(resnet18, verbosity_level=2, enable_cuda=False)

    # When setting a Classifers base dataset, it automatically configures the Classifier to work with all classes in
    # the base dataset. When no other dataset specified for training or testing, the classier will use its base dataset
    base_dataset = gem.ConcurrentPickledDicomoTaskSplitter(base_path=train_dataset, dicomo_fields=['tensor', 'bpe'])
    net.set_base_dataset(base_dataset)
    net.save(classifier_path)


def demo_train_classifier():
    # Load a classifier from a file
    net = gem.Classifier.from_pickle(classifier_path)

    # Train the classifier
    # net.set_trainable_layers([("all", True)]) # by default all layers are trainable
    # net.set_device(enable_cuda=False)
    # net.train(get_data_set(train_data_set_path), num_workers=6, epochs=1, pin_memory=True)
    net.train(epochs=20)
    net.save(trained_classifier_path)


def demo_evaluate_classifier():
    net = gem.Classifier.from_pickle(trained_classifier_path)
    net.evaluate(get_data_set(eval_data_set_path), num_workers=0, pin_memory=True)


def demo_create_dicomo_dataset():
    data_origin = '/mnt/data2/pukkaj/teach/study/PJ_TEACH/PJ_RESEARCH/'
    data_destination = '/home/nheinen/gemicai/dicom_objects/DX/'
    gem.compress_dicom_files(data_origin, data_destination, modalities=['DX'])


def demo_things_we_should_fix():
    # FIXME: Exception handling in the data_iterators (Not extremely important rn but something we should consider)
    #  This raises a StopIteration exception, should be a FileNotFound or something like that.
    dataset = get_data_set('wrong_path')
    data_loader = torch.utils.data.DataLoader(dataset, 4)
    for data in data_loader:
        pass


# TODO: Perhaps generalize this as well for the library, or make a function 'get_dicomo_dataset()'
def get_data_set(data_directory, object_fields=['tensor', 'bpe'], use_pds=False):
    transform = gem.torchvision.transforms.Compose([
        gem.torchvision.transforms.ToPILImage(),
        gem.torchvision.transforms.Grayscale(3),
        gem.torchvision.transforms.ToTensor()
    ])
    # We don't need a transform everytime right? The data has been transformed, and stored as tensor in the dicomo files
    transform = None

    if use_pds:
        # while creating PickleDataSet we pass a path to a pickle that hold the data
        # and a list of the fields that we want to extract from the dicomo object
        return gem.iterators.PickledDicomoDataSet(data_directory, ['tensor', 'bpe'], transform)
    else:
        return gem.iterators.ConcurrentPickledDicomoTaskSplitter(data_directory, ['tensor', 'bpe'], transform)


# this has to wrap the code we call
# you can say thank you to how python implements multithreading
# and yes it has to be here and not in the Classifier.py
if __name__ == '__main__':
    demo_initialize_classifier()
    demo_train_classifier()
#   demo_evaluate_classifier()
# demo_create_dicomo_dataset()
