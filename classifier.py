import pickle
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchsummary import summary
import PickleDataSet
import os

import dicom_utilities as du

# Input shape of the tensors used by the classifier, this is (3, 244)
input_shape = (3, 244, 244)

# All possible classes, #TODO @Niek get complete list
labels = ['Thorax PA', 'Thorax AP', 'Thorax AX', 'etc']

resnet18 = models.resnet18(pretrained=True)


class Classifier:
    def __init__(self, base_model: nn.Module):
        # Sets model of the classifier and edits its last layer.
        self.model = base_model
        self.model.fc = nn.Linear(self.model.fc.in_features, len(labels))

        # Default setting for training, can be overwirten in train() if save_as_default=True
        self.epochs = 20
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        # Save location, will be stored after setting it once with save()
        self.file_path = None

    def set_trainable_layers(self, layers, boolean):
        for name, param in self.model.named_parameters():
            name = '.'.join(name.split('.')[:-1])
            if layers == 'all' or name in layers:
                param.requires_grad = boolean

    def summary(self):
        summary(self.model, input_shape)

    # TODO implement this function with torch.utils.data.DataLoader object, we can then classify images easily
    # with self.model(images) like in https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    def classifiy(self, dicom_file_path):
        tensor, label = du.dicom_get_tensor_and_label(dicom_file_path)
        return self.model(tensor)

    def eval(self, verbosity=0):
        # TODO write function that evaluates self.model
        pass

    def train(self, epochs=None, loss_function=None, optimizer=None, verbosity=0, save_as_default=False):
        if epochs is None:
            epochs = self.epochs
        if loss_function is None:
            loss_function = self.loss_function
        if optimizer is None:
            optimizer = self.optimizer
        if save_as_default:
            self.epochs = epochs
            self.loss_function = loss_function
            self.optimizer = optimizer

        for epoch in range(epochs):
            running_loss = 0.0
            dataloader = get_data_loader()
            for i, data in enumerate(dataloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if verbosity >= 1 and i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

    # save classifier object to .pkl file, can be retrieved with load_classifier()
    def save(self, file_path=None):
        if file_path is None:
            file_path = self.file_path
        else:
            self.file_path = file_path
        with open(file_path, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)


# Loads classifier objcet from .pkl file
def load_classifier(pkl_file_path):
    with open(pkl_file_path, 'rb') as input:
        cf = pickle.load(input)
        assert isinstance(cf, Classifier), 'Not a valid Classifier'
        print('Loaded classifier from {} as {}'.format(cf.file_path, cf))
        return cf


#os.path.join makes a platform dependent path (so both linux and windows works)
def get_data_loader(data_directory=os.path.join('examples', 'compressed', 'CT', '000001.gz'), batch_size=4):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Grayscale(3),
        torchvision.transforms.ToTensor()
    ])

    # while creating PickleDataSet we pass a path to a pickle that hold the data
    # and a list of the fields that we want to extract from the dicomo object
    pickle_iter = PickleDataSet.PickleDataSet(data_directory, ['tensor', 'bpe'], transform)

    # since we use a file with arbitrary number of dicomo objects we cannot parallelize loading data.
    # On the bright side we load only objects we currently need (batch_size) into memory
    return torch.utils.data.DataLoader(pickle_iter, batch_size, shuffle=False, num_workers=0)


# Demo code

# create a pickle to use as a data source
# origin = os.path.join('examples', 'dicom', 'CT')
# destination = os.path.join('examples', 'compressed', 'CT/')
# PickleDataSet.dicomo.compress_dicom_files(origin, destination)

classifier = Classifier(resnet18)
dataloader = get_data_loader()


# fetch a new batch
for tensor, bpe in dataloader:
    # in this case each batch contains 4 tensors and labels
    # trying to feed the input to the model
    classifier.model(tensor)
    print("IT DID NOT CRASH")

    # print the tensors and labels
    # WARNING!
    # only works if PickleDataSet in the get_data_loader function does not takes transform
    # variable as a parameter, Grayscale(3) somehow fucks it up
    # PickleDataSet.print_labels_and_display_images(tensor, bpe)

# print('Classifier summaray per layer, keras style')
# classifier.summary()
# print('Classifier summaray per layer')
# print(classifier.model)
#
# classifier.set_trainable_layers('all', False)
# classifier.summary()

# pkl = 'classifiers/resnet18.pkl'
# classifier.save(pkl)
# del classifier
# classifier = load_classifier(pkl)
