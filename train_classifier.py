import torchvision.models as models
import gemicai as gem
import os


def demo_initialize_classifier():
    # Use resnet 18 as the base model for our new classifier
    resnet18 = models.resnet18(pretrained=True)
    net = gem.Classifier(resnet18, enable_cuda=False)

    # Setting data_loader of the network, this takes a while as it automaticly determines all classes.
    # Set verbosity to 1 if you like print statements to the terminal.
    #/home/nheinen/gemicai/dicom_objects/DX/'
    path = os.path.join("examples", "zip", "CT")
    net.set_data_loader('examples/compressed/DX/', verbosity=1)

    # Saves the classifier to a file, this way you don't have to rebuild the whole classifier everytime.
    net.save('classifiers/dx_bpe.pkl')


def demo_train_classifier():
    # Load a classifier from a file
    net = gem.load_classifier('classifiers/dx_bpe.pkl', verbosity=1)

    # Train the classifier
    net.train(epochs=2, verbosity=1)
    net.save('classifiers/dx_bpe_trained.pkl')


def demo_evaluate_classifier():
    net = gem.load_classifier('classifiers/dx_bpe_trained.pkl')

    # Evaluate the classifier, specify the directory of what images it should be evaluated with.
    evaluation_data = '/home/nheinen/gemicai/dicom_objects/DX/'
    net.evaluate('examples/compressed/DX/', verbosity=1)


def demo_create_dicomo_dataset():
    data_origin = '/mnt/data2/pukkaj/teach/study/PJ_TEACH/PJ_RESEARCH/'
    data_destination = '/home/nheinen/gemicai/dicom_objects/DX/'
    gem.compress_dicom_files(data_origin, data_destination, modalities=['DX'])

# demo_initialize_classifier()
# demo_train_classifier()
# demo_evaluate_classifier()
demo_create_dicomo_dataset()
