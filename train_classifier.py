import torchvision.models as models
import gemicai.dicomo as dicomo
import gemicai as gem
import os

def demo_initialize_classifier():
    # Use resnet 18 as the base model for our new classifier
    resnet18 = models.resnet18(pretrained=True)
    net = gem.Classifier(resnet18, enable_cuda=True)

    # Setting data_loader of the network, this takes a while as it automaticly determines all classes.
    # Set verbosity to 1 if you like print statements to the terminal.
    #/home/nheinen/gemicai/dicom_objects/DX/'
    path = os.path.join("examples", "zip", "CT")
    net.set_data_loader(path, verbosity=1)

    # Saves the classifier to a file, this way you don't have to rebuild the whole classifier everytime.
    net.save('classifiers/dx_bpe.pkl')


def demo_train_classifier():
    # Load a classifier from a file
    net = gem.load_classifier('classifiers/dx_bpe.pkl', verbosity=1)

    # Train the classifier
    net.train(epochs=20, verbosity=1)
    net.save('classifiers/dx_bpe_trained.pkl')


def demo_evaluate_classifier():
    net = gem.load_classifier('classifiers/dx_bpe_trained.pkl')

    # Evaluate the classifier, specify the directory of what images it should be evaluated with.
    net.evaluate('/home/nheinen/gemicai/dicom_objects/DX/', verbosity=1)

#dicomo.compress_dicom_files("examples/dicom/CT", "examples/gzip/CT", objects_per_file=25)

path = os.path.join("examples", "gzip", "CT")
loader = gem.get_data_loader(path)

for tensors, labels in loader:
    print(labels)

#demo_initialize_classifier()
# demo_train_classifier()
# demo_evaluate_classifier()
