import pickle
import torch
import torch.nn as nn
import torchvision
from torchsummary import summary
import PickleDataSet
from dicomo import LabelCounter


class Classifier:
    def __init__(self, base_model: nn.Module):
        # Sets base model of the classifier
        self.model = base_model

        # Default setting for training, can be overwirten in train() if save_as_default=True
        self.epochs = 20
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.batch_size = 4

        # Save location, will be stored after setting it once with save()
        self.file_path = None

        # Data loader will be set to None, has to bet set with set_data_loader() in order to train the classifier.
        self.data_loader = None
        self.train_directory = None
        self.classes = None

        # Input shape of the tensors used by the classifier, only needed for keras like model summary
        self.input_shape = (3, 244, 244)

    def set_trainable_layers(self, layers, boolean):
        for name, param in self.model.named_parameters():
            name = '.'.join(name.split('.')[:-1])
            if layers == 'all' or name in layers:
                param.requires_grad = boolean

    def summary(self):
        summary(self.model, self.input_shape)

    def evaluate(self, evaluation_directory, verbosity=0):
        # puts model in evaluation mode.
        self.model.eval()
        correct, total = 0, 0
        testloader = get_data_loader(evaluation_directory, batch_size=self.batch_size)
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('Total: {} -- Correct: {} -- Accuracy: {}%'.format(total, correct, round(100 * correct / total, 2)))

    def set_data_loader(self, train_directory, verbosity=0, determine_classes=True):
        self.data_loader = get_data_loader(data_directory=train_directory, batch_size=self.batch_size)
        self.train_directory = train_directory
        # This automatically determines all classes within the training data, and alters the classifer accordingly
        if determine_classes:
            cnt = LabelCounter()
            for i, data in enumerate(self.data_loader):
                for label in data[1]:
                    cnt.update(label)
            if verbosity >= 1:
                cnt.print()
            self.classes = list(cnt.dic.keys())
            self.model.fc = nn.Linear(self.model.fc.in_features, len(self.classes))

    def train(self, epochs=None, loss_function=None, optimizer=None, verbosity=0, save_as_default=False):
        # Puts model in training mode.
        self.model.train()
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
            for i, data in enumerate(self.data_loader):
                # get the inputs; data is a list of [tensors, labels]
                tensors, labels = data

                # labels returned by the classifier are strings, we need to convert this to an int
                labels = torch.tensor([self.classes.index(label) for label in labels])

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(tensors)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if verbosity >= 1 and i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

    # save classifier object to .pkl file, can be retrieved with load_classifier()
    def save(self, file_path=None):
        if file_path is None:
            file_path = self.file_path
        else:
            self.file_path = file_path
        # You can't store a generator object.
        self.data_loader = None
        with open(file_path, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)


# Loads classifier objcet from .pkl file
def load_classifier(pkl_file_path, verbosity=0):
    with open(pkl_file_path, 'rb') as input:
        cf = pickle.load(input)
        assert isinstance(cf, Classifier), 'Not a valid Classifier'
        cf.set_data_loader(cf.train_directory, determine_classes=False)
        if verbosity >= 1:
            print('Succesfully loaded classifier with classes: {}'.format(cf.classes))
        return cf


# os.path.join makes a platform dependent path (so both linux and windows works)
# =os.path.join('examples', 'compressed', 'CT', '000001.gz')
def get_data_loader(data_directory, use_pds=False, batch_size=4):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Grayscale(3),
        torchvision.transforms.ToTensor()
    ])

    if use_pds:
        # while creating PickleDataSet we pass a path to a pickle that hold the data
        # and a list of the fields that we want to extract from the dicomo object
        pickle_iter = PickleDataSet.PickleDataSet(data_directory, ['tensor', 'bpe'], transform)
    else:
        pickle_iter = PickleDataSet.PickleDataFolder(data_directory, ['tensor', 'bpe'], transform)

    # since we use a file with arbitrary number of dicomo objects we cannot parallelize loading data.
    # On the bright side we load only objects we currently need (batch_size) into memory
    return torch.utils.data.DataLoader(pickle_iter, batch_size, shuffle=False, num_workers=0)
