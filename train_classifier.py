from gemicai.data_iterators import DicomoDataset as DataSet
import torchvision.models as models
import gemicai as gem
import torch
import os

train_data_set_path = os.path.join("examples", "gzip", "CT")
eval_data_set_path = os.path.join("examples", "gzip", "CT")
classifier_path = os.path.join("classifiers", "dx_bpe.pkl")
trained_classifier_path = os.path.join("classifiers", "dx_bpe_trained.pkl")

train_dataset = '/mnt/SharedStor/datasets/dx/train/'
test_dataset = '/mnt/SharedStor/datasets/dx/test/'
# classifier_path = '/mnt/SharedStor/classifiers/dx_bpe.pkl'

def demo_initialize_classifier():
    # Use resnet 18 as the base model for our new classifier
    resnet18 = models.resnet18(pretrained=True)
    net = gem.Classifier(resnet18, verbosity_level=2, enable_cuda=False)

    # When setting a Classifers base dataset, it automatically configures the Classifier to work with all classes in
    # the base dataset. When no other dataset specified for training or testing, the classier will use its base dataset
    base_dataset = DataSet.get_dicomo_data_set(train_data_set_path, ['tensor', 'bpe'])
    net.set_base_dataset(base_dataset)
    net.save(classifier_path)


def demo_train_classifier():
    # Load a classifier from a file
    net = gem.Classifier.from_pickle(classifier_path)

    # Train the classifier
    # net.set_trainable_layers([("all", True)]) # by default all layers are trainable
    # net.set_device(enable_cuda=False)
    # net.train(get_data_set(train_data_set_path), num_workers=6, epochs=1, pin_memory=True)
    net.train(num_workers=6, epochs=20, pin_memory=True)
    net.save(trained_classifier_path)


def demo_evaluate_classifier():
    net = gem.Classifier.from_pickle(trained_classifier_path)
    net.evaluate(num_workers=0, pin_memory=True)


def demo_create_dicomo_dataset():
    data_origin = '/mnt/data2/pukkaj/teach/study/PJ_TEACH/PJ_RESEARCH/'
    data_destination = '/home/nheinen/gemicai/dicom_objects/DX/'
    gem.compress_dicom_files(data_origin, data_destination, modalities=['DX'])


def demo_things_we_should_fix():
    # FIXME: Exception handling in the data_iterators (Not extremely important rn but something we should consider)
    #  This raises a StopIteration exception, should be a FileNotFound or something like that.
    dataset = DataSet.get_dicomo_data_set('wrong_path')
    data_loader = torch.utils.data.DataLoader(dataset, 4)
    for data in data_loader:
        pass


# TODO this should work but doesnt @Mateusz I think you might have solved this issue before. Could you take a look at it?
def demo_dicomo_dataset():
    dataset = gem.DicomoDataset.from_directory(train_data_set_path)  # ['tensor', 'bpe']
    for data in dataset:
        print(data)


# this has to wrap the code we call
# you can say thank you to how python implements multithreading
# and yes it has to be here and not in the Classifier.py
if __name__ == '__main__':
    demo_initialize_classifier()
    demo_train_classifier()
#    demo_dicomo_dataset()
    demo_evaluate_classifier()
# demo_create_dicomo_dataset()
