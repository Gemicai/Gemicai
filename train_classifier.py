import classifier
import torchvision.models as models


def demo_initialize_classifier():
    # Use resnet 18 as the base model for our new classifier
    resnet18 = models.resnet18(pretrained=True)
    net = classifier.Classifier(resnet18)

    # Setting data_loader of the network, this takes a while as it automaticly determines all classes.
    # Set verbosity to 1 if you like print statements to the terminal.
    net.set_data_loader('/home/nheinen/gemicai/dicom_objects/DX/', verbosity=1)

    # Saves the classifier to a file, this way you don't have to rebuild the whole classifier everytime.
    net.save('classifiers/dx_bpe.pkl')


def demo_train_classifier():
    # Load a classifier from a file
    net = classifier.load_classifier('classifiers/dx_bpe.pkl')

    # Train the classifier
    net.train(epochs=20, verbosity=1)

    # Evaluate the classifier, specify the directory of what images it should be evaluated with.
    net.evaluate('/home/nheinen/gemicai/dicom_objects/DX/', verbosity=1)
    net.save('classifiers/dx_bpe_trained.pkl')


# demo_initialize_classifier()
demo_train_classifier()
